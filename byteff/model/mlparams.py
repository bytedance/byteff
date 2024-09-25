# -----  ByteFF: ByteDance Force Field -----
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from collections import defaultdict

import torch
from torch import Tensor, nn
from torch_geometric.data import Data

from byteff.model import layers
from byteff.utils.definitions import ParamTerm, TopoParams, TopoTerm

logger = logging.getLogger(__name__)


class ParamOutOfRangeError(Exception):
    pass


class CustomClamp(torch.autograd.Function):  # pylint: disable=abstract-method

    @staticmethod
    def forward(ctx, _input, _min=None, _max=None):
        return _input.clamp(min=_min, max=_max)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone(), None, None


custom_clamp = CustomClamp.apply


class MLParams(nn.Module):

    def __init__(self,
                 feature_layer: dict,
                 gnn_layer: dict,
                 output_layer: dict,
                 param_range: dict[str, list] = None,
                 use_nonbonded_label: bool = False):
        '''
        Model for 
            1. extract atomic features
            2. graph model
            3. JanossyPooling to output permutation invarient classical FF K parameters
                https://arxiv.org/abs/1811.01900

        @param:
            feature_layer: feature_layer type and kwargs
            gnn_layer: gnn_layer type and kwargs
            output_layer: output_layer type and kwargs
            param_range: upper and lower bounds for parameters
            use_nonbonded_label: use nonbonded parameters saved in data instead of prediction
        '''

        super().__init__()
        self.use_nonbonded_label = use_nonbonded_label
        self.param_range = {}
        if param_range is not None:
            for k, v in param_range.items():
                term = getattr(ParamTerm, k)
                assert isinstance(v, list) and len(v) == 2
                assert v[0] is None or isinstance(v[0], float)
                assert v[1] is None or isinstance(v[1], float)
                self.param_range[term] = v.copy()

        feature_layer = feature_layer.copy()
        feature_layer_type = feature_layer.pop('type', 'FeatureLayer')
        self.feature_layer: layers.FeatureLayer = getattr(layers, feature_layer_type)(**feature_layer)

        gnn_layer = gnn_layer.copy()
        gnn_layer_type = gnn_layer.pop('type', 'GINLayer')
        self.gnn_layer: layers.GNNLayerBase = getattr(layers, gnn_layer_type)(self.feature_layer.node_dim,
                                                                              self.feature_layer.edge_dim, **gnn_layer)

        output_layer = output_layer.copy()
        output_layer_type = output_layer.pop('type', 'OutputLayer')
        self.output_layer: layers.OutputLayer = getattr(layers, output_layer_type)(node_dim=self.gnn_layer.node_out_dim,
                                                                                   edge_dim=self.gnn_layer.edge_out_dim,
                                                                                   **output_layer)
        self._reset_parameter()

    def _reset_parameter(self):
        self.feature_layer._reset_parameter()
        self.gnn_layer._reset_parameter()
        self.output_layer._reset_parameter()

    def _apply_params_range(self, params: dict[ParamTerm, Tensor], raise_out_of_range=False) -> dict[ParamTerm, Tensor]:
        for p_term in params:
            if p_term in self.param_range:
                y = params[p_term]
                if raise_out_of_range:
                    if (y > self.param_range[p_term][1]).any() or (y < self.param_range[p_term][0]).any():
                        raise ParamOutOfRangeError(f'param out of range, {p_term}')
                params[p_term] = custom_clamp(y, *self.param_range[p_term])
        return params

    def _make_topo_params(self, graph: Data, params: dict[ParamTerm, Tensor]) -> TopoParams:
        topo_params = TopoParams()

        for term in TopoTerm:
            if hasattr(graph, term.name + '_index'):
                topo_params[term] = getattr(graph, term.name + '_index').T
                topo_params.counts[term] = getattr(graph, 'n' + term.name)
            else:
                logger.warning(f'skip {term}')
        for k, v in params.items():
            topo_params[k] = v
        if self.use_nonbonded_label:
            for term in [ParamTerm.Charge, ParamTerm.Sigma, ParamTerm.Epsilon]:
                topo_params[term] = getattr(graph, term.name + '_label')

        if not self.output_layer.use_improper_torsion:
            b, bk = topo_params[TopoTerm.Bond], topo_params[ParamTerm.Bond_k]
            topo_params[TopoTerm.ImproperTorsion] = torch.zeros((0, 4), dtype=b.dtype, device=b.device)
            topo_params[ParamTerm.ImproperTorsion_k] = torch.zeros((0, 1), dtype=bk.dtype, device=bk.device)

        return topo_params

    def forward(self, graph: Data, raise_out_of_range=False) -> TopoParams:
        '''Predict forcefield parameters
        @param:
            graph: Data:
        @return:
            TopoParams, containing topology and predicted parameters
        '''

        node_feature, edge_features = self.feature_layer(graph)

        x_h, e_h = self.gnn_layer(graph, node_feature, edge_features)

        params = self.output_layer(x_h, e_h, graph)

        if self.param_range:
            params = self._apply_params_range(params, raise_out_of_range)

        # mask linear proper
        if hasattr(graph, "ProperTorsion_linear_mask"):
            params[ParamTerm.ProperTorsion_k] *= graph.ProperTorsion_linear_mask.unsqueeze(-1)

        topo_params = self._make_topo_params(graph, params)

        return topo_params


class EnsembleModel:

    def __init__(self, models: list[MLParams]):
        self.models = models

        # calculate std and rstd when calling
        self.std = None
        self.rstd = None

    @torch.no_grad()
    def __call__(self, *args, **kwds):
        params = defaultdict(list)
        for model in self.models:
            tp: TopoParams = model(*args, **kwds)
            for p in ParamTerm:
                params[p].append(tp[p])
        for k, v in params.items():
            ps = torch.concat([p.unsqueeze(0) for p in v], dim=0)
            mean_ps = ps.mean(dim=0)
            std_ps = ps.std(dim=0)
            rstd_ps = std_ps / torch.where(mean_ps > 1.0, mean_ps, 1.0)
            tp[k] = mean_ps
            tp.uncertainty[k] = rstd_ps
        return tp

    def eval(self):
        for model in self.models:
            model.eval()
