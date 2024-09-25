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

import torch
from torch import Tensor, nn
from torch_geometric.data import Data
from torch_geometric.nn import MLP
from torch_geometric.utils import scatter

from byteff.model.gnn import EGT, GAT, GIN, GINE, GT, BasicGNN
from byteff.utils.definitions import (MAX_FORMAL_CHARGE, MAX_RING_SIZE, MAX_TOTAL_CHARGE, PROPERTORSION_TERMS,
                                      BondOrder, ParamData, ParamTerm, TopoParams, TopoTerm, supported_atomic_number)


def to_global_edge_idx(edge_idx: Tensor, nedges: Tensor, counts: Tensor):
    """ get the edge_idx after batch
        nedge is the number of edges in each molecule, 
        counts is the number of edge_idx in each molecule
    """
    edge_idx = edge_idx.long()
    nedges = nedges.long()
    counts = counts.long()
    nedges_cumsum = torch.cumsum(nedges, 0)  # [batch_size]
    nedges_cumsum = torch.concat((torch.tensor([0], device=nedges.device, dtype=nedges.dtype), nedges_cumsum[:-1]),
                                 dim=0)  # [batch_size]
    size = (-1,) + (1,) * (edge_idx.dim() - 1)
    edge_idx += torch.repeat_interleave(nedges_cumsum, counts).view(size)  # [nterm, ...]
    return edge_idx


def equi_features(features: Tensor, equi_index: Tensor) -> Tensor:
    equi_value = scatter(features, equi_index, 0, reduce='mean')
    return equi_value[equi_index]


class FeatureLayer(nn.Module):
    """ atom features and/or bond features
    """

    def __init__(
            self,
            atom_embedding_dim=16,
            tot_chg_embedding_dim=16,
            ring_con_embedding_dim=0,
            min_ring_size_embedding_dim=0,
            scale_grad_by_freq=False,
            node_mlp_dims=(32, 32, 3),
            act='gelu',
    ) -> None:
        super().__init__()

        self.atom_embedding_dim = atom_embedding_dim
        self.atom_embedding = nn.Embedding(num_embeddings=len(supported_atomic_number),
                                           embedding_dim=atom_embedding_dim,
                                           scale_grad_by_freq=scale_grad_by_freq)
        self.tot_chg_embedding_dim = tot_chg_embedding_dim
        self.tot_chg_embedding = nn.Embedding(num_embeddings=2 * MAX_TOTAL_CHARGE + 1,
                                              embedding_dim=tot_chg_embedding_dim,
                                              scale_grad_by_freq=scale_grad_by_freq)
        self.ring_con_embedding_dim = ring_con_embedding_dim
        self.ring_con_embedding = nn.Embedding(num_embeddings=8,
                                               embedding_dim=ring_con_embedding_dim,
                                               scale_grad_by_freq=scale_grad_by_freq)
        self.min_ring_size_embedding_dim = min_ring_size_embedding_dim
        self.min_ring_size_embedding = nn.Embedding(num_embeddings=MAX_RING_SIZE + 1,
                                                    embedding_dim=min_ring_size_embedding_dim,
                                                    scale_grad_by_freq=scale_grad_by_freq)

        # node mlp
        self.node_mlp = MLP(in_channels=self.raw_node_dim,
                            hidden_channels=node_mlp_dims[0],
                            out_channels=node_mlp_dims[1],
                            num_layers=node_mlp_dims[2],
                            norm=None,
                            act=act,
                            plain_last=False)
        self.node_dim = node_mlp_dims[1]
        self.edge_dim = None

    @property
    def raw_node_dim(self) -> int:
        """dim for raw node feature"""
        return self.atom_embedding_dim + self.tot_chg_embedding_dim + self.ring_con_embedding_dim + self.min_ring_size_embedding_dim

    def _reset_parameter(self):
        '''Reset parameters using kaiming_uniform (default)'''
        self.atom_embedding.reset_parameters()
        if isinstance(self.tot_chg_embedding, nn.Embedding):
            self.tot_chg_embedding.reset_parameters()
        self.ring_con_embedding.reset_parameters()
        self.min_ring_size_embedding.reset_parameters()
        self.node_mlp.reset_parameters()

    def get_node_features(self, graph: Data) -> Tensor:
        x = graph.x  # [natoms, 2]
        x = x.long()
        embeddings = []
        embeddings.append(self.atom_embedding(x[:, 0]))  # [natoms, atom_embedding_dim]
        embeddings.append(
            self.tot_chg_embedding(torch.clamp(x[:, 1] + MAX_TOTAL_CHARGE, min=0,
                                               max=2 * MAX_TOTAL_CHARGE)))  # [natoms, tot_chg_embedding_dim]
        embeddings.append(self.ring_con_embedding(x[:, 3]))  # [natoms, ring_con_embedding_dim]
        embeddings.append(self.min_ring_size_embedding(x[:, 4]))  # [natoms, min_ring_size_embedding_dim]
        node_features = torch.concat(embeddings, dim=-1)
        node_features = self.node_mlp(node_features)
        node_features = equi_features(node_features, graph.equi_index.long())
        return node_features

    def get_edge_features(self, graph: Data):
        return None

    def forward(self, graph: Data) -> tuple[Tensor, Tensor]:
        """ return node and edge features
        """
        node_features = self.get_node_features(graph)
        edge_features = self.get_edge_features(graph)  # pylint: disable=assignment-from-none
        return node_features, edge_features


class ExtendedFeature(FeatureLayer):

    def __init__(
            self,
            fm_chg_embedding_dim=8,
            bond_ring_embedding_dim=8,
            bond_order_embedding_dim=8,
            scale_grad_by_freq=False,
            edge_mlp_dims=(32, 32, 3),
            act='gelu',
            **kwargs,
    ) -> None:

        self.bond_ring_embedding_dim = bond_ring_embedding_dim
        self.bond_order_embedding_dim = bond_order_embedding_dim
        self.fm_chg_embedding_dim = fm_chg_embedding_dim

        super().__init__(**kwargs, scale_grad_by_freq=scale_grad_by_freq, act=act)
        self.bond_ring_embedding = nn.Embedding(num_embeddings=2,
                                                embedding_dim=bond_ring_embedding_dim,
                                                scale_grad_by_freq=scale_grad_by_freq)
        self.bond_order_embedding = nn.Embedding(num_embeddings=len(BondOrder),
                                                 embedding_dim=bond_order_embedding_dim,
                                                 scale_grad_by_freq=scale_grad_by_freq)
        self.fm_chg_embedding = nn.Embedding(num_embeddings=MAX_FORMAL_CHARGE * 2 + 1,
                                             embedding_dim=fm_chg_embedding_dim,
                                             scale_grad_by_freq=scale_grad_by_freq)

        # edge mlp
        self.edge_mlp = MLP(in_channels=self.raw_edge_dim,
                            hidden_channels=edge_mlp_dims[0],
                            out_channels=edge_mlp_dims[1],
                            num_layers=edge_mlp_dims[2],
                            norm=None,
                            act=act,
                            plain_last=False)
        self.edge_dim = edge_mlp_dims[1]

    def _reset_parameter(self):
        super()._reset_parameter()
        self.bond_ring_embedding.reset_parameters()
        self.bond_order_embedding.reset_parameters()
        self.fm_chg_embedding.reset_parameters()
        self.edge_mlp.reset_parameters()

    @property
    def raw_node_dim(self) -> int:
        """dim for raw node feature"""
        return self.atom_embedding_dim + self.tot_chg_embedding_dim + self.fm_chg_embedding_dim \
                + self.ring_con_embedding_dim + self.min_ring_size_embedding_dim

    @property
    def raw_edge_dim(self) -> int:
        """dim for raw edge feature"""
        return self.bond_order_embedding_dim + self.bond_ring_embedding_dim

    def get_node_features(self, graph: Data):
        x = graph.x  # [natoms, 2]
        x = x.long()
        embeddings = []
        embeddings.append(self.atom_embedding(x[:, 0]))  # [natoms, atom_embedding_dim]
        embeddings.append(
            self.tot_chg_embedding(torch.clamp(x[:, 1] + MAX_TOTAL_CHARGE, min=0,
                                               max=2 * MAX_TOTAL_CHARGE)))  # [natoms, tot_chg_embedding_dim]
        embeddings.append(
            self.fm_chg_embedding(torch.clamp(x[:, 2] + MAX_FORMAL_CHARGE, min=0,
                                              max=2 * MAX_FORMAL_CHARGE)))  # [natoms, fm_chg_embedding_dim]
        embeddings.append(self.ring_con_embedding(x[:, 3]))  # [natoms, ring_con_embedding_dim]
        embeddings.append(self.min_ring_size_embedding(x[:, 4]))  # [natoms, min_ring_size_embedding_dim]
        node_features = torch.concat(embeddings, dim=-1)
        node_features = self.node_mlp(node_features)
        node_features = equi_features(node_features, graph.equi_index.long())
        return node_features

    def get_edge_features(self, graph: Data):
        edge_feat = getattr(graph, "edge_features")
        edge_ring = self.bond_ring_embedding(edge_feat[:, 0])
        edge_order = self.bond_order_embedding(edge_feat[:, 1])
        edge_features = torch.concat([edge_ring, edge_order], dim=-1)
        edge_features = self.edge_mlp(edge_features)
        equi_edge_idx = getattr(graph, "equi_edge_idx")
        nedges = getattr(graph, "nBond") * 2
        equi_edge_idx = to_global_edge_idx(equi_edge_idx, nedges, nedges)
        edge_features = equi_features(edge_features, equi_edge_idx)
        return edge_features


class LocalFeature(ExtendedFeature):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        # remove total charge embedding
        self.tot_chg_embedding = None

    @property
    def raw_node_dim(self) -> int:
        """dim for raw node feature"""
        return self.atom_embedding_dim + self.fm_chg_embedding_dim \
                + self.ring_con_embedding_dim + self.min_ring_size_embedding_dim

    def get_node_features(self, graph: Data):
        x = graph.x  # [natoms, 2]
        x = x.long()
        embeddings = []
        embeddings.append(self.atom_embedding(x[:, 0]))  # [natoms, atom_embedding_dim]
        embeddings.append(
            self.fm_chg_embedding(torch.clamp(x[:, 2] + MAX_FORMAL_CHARGE, min=0,
                                              max=2 * MAX_FORMAL_CHARGE)))  # [natoms, fm_chg_embedding_dim]
        embeddings.append(self.ring_con_embedding(x[:, 3]))  # [natoms, ring_con_embedding_dim]
        embeddings.append(self.min_ring_size_embedding(x[:, 4]))  # [natoms, min_ring_size_embedding_dim]
        node_features = torch.concat(embeddings, dim=-1)
        node_features = self.node_mlp(node_features)
        node_features = equi_features(node_features, graph.equi_index.long())
        return node_features


class GNNLayerBase(nn.Module):
    """ base model for graph"""

    def __init__(self, node_in_dim, edge_in_dim, gnn_dims=(32, 32, 3), act='gelu', jk=None, **kwargs):
        super().__init__()

        self.node_in_dim = node_in_dim
        self.edge_in_dim = edge_in_dim

        self.gnn, self.node_out_dim = self._init_gnn(gnn_dims, act, jk, **kwargs)
        self.edge_out_dim = edge_in_dim

    def _init_gnn(self, gnn_dims, act, jk, **kwargs) -> tuple[BasicGNN, int]:
        raise NotImplementedError

    def _reset_parameter(self):
        '''Reset parameters using kaiming_uniform (default)'''
        self.gnn.reset_parameters()

    def forward(self, graph: Data, x_h: Tensor, e_h: Tensor):

        edge_index = graph.edge_index  # [2, nbonds * 2]
        edge_index = edge_index.long()

        if self.gnn.supports_edge_update:
            x_h, e_h = self.gnn(x_h, edge_index, e_h)
        else:
            x_h = self.gnn(x_h, edge_index, e_h)

        # average bidirectional edge
        equi_edge_idx = getattr(graph, "equi_edge_idx")
        nedges = getattr(graph, "nBond") * 2
        equi_edge_idx = to_global_edge_idx(equi_edge_idx, nedges, nedges)
        e_h = equi_features(e_h, equi_edge_idx)

        return x_h, e_h


class GINLayer(GNNLayerBase):

    def _init_gnn(self, gnn_dims, act, jk, **kwargs):
        gnn = GIN(in_channels=self.node_in_dim,
                  hidden_channels=gnn_dims[0],
                  out_channels=gnn_dims[1],
                  num_layers=gnn_dims[2],
                  act=act,
                  jk=jk,
                  **kwargs)
        return gnn, gnn_dims[1]


class GINELayer(GNNLayerBase):

    def _init_gnn(self, gnn_dims, act, jk, **kwargs):
        gnn = GINE(in_channels=self.node_in_dim,
                   hidden_channels=gnn_dims[0],
                   out_channels=gnn_dims[1],
                   num_layers=gnn_dims[2],
                   act=act,
                   jk=jk,
                   edge_dim=self.edge_in_dim,
                   **kwargs)
        return gnn, gnn_dims[1]


class GATLayer(GNNLayerBase):

    def _init_gnn(self, gnn_dims, act, jk, **kwargs):
        gnn = GAT(in_channels=self.node_in_dim,
                  hidden_channels=gnn_dims[0],
                  out_channels=gnn_dims[1],
                  num_layers=gnn_dims[2],
                  act=act,
                  jk=jk,
                  edge_dim=self.edge_in_dim,
                  **kwargs)
        return gnn, gnn_dims[1]


class GTLayer(GNNLayerBase):

    def _init_gnn(self, gnn_dims, act, jk, **kwargs):
        gnn = GT(in_channels=self.node_in_dim,
                 hidden_channels=gnn_dims[0],
                 out_channels=gnn_dims[1],
                 num_layers=gnn_dims[2],
                 act=act,
                 jk=jk,
                 edge_dim=self.edge_in_dim,
                 **kwargs)
        return gnn, gnn_dims[1]


class EGTLayer(GNNLayerBase):

    def _init_gnn(self, gnn_dims, act, jk, **kwargs):
        gnn = EGT(
            in_channels=self.node_in_dim,
            hidden_channels=gnn_dims[0],
            out_channels=gnn_dims[1],
            num_layers=gnn_dims[2],
            act=act,
            jk=jk,
            #   edge_dim=self.edge_in_dim,
            **kwargs)
        return gnn, gnn_dims[1]


class OutputLayer(nn.Module):
    """ post_mlp -> symmetry-preserving pooling -> out_mlp
    """

    def __init__(self,
                 node_dim: int,
                 edge_dim: int,
                 post_mlp_dims=(32, 32, 3),
                 out_mlp_dims=(32, 3),
                 act='gelu',
                 tanh_output=15.,
                 post_mlp_dims_proper=None,
                 out_mlp_dims_proper=None,
                 use_improper_torsion=True) -> None:

        super().__init__()

        self.use_improper_torsion = use_improper_torsion
        self.param_shapes = {
            ParamTerm.Sigma: (1, 1),
            ParamTerm.Epsilon: (1, 1),
            ParamTerm.Charge: (1, 1),
            ParamTerm.Bond_k: (2, 1),
            ParamTerm.Bond_length: (2, 1),
            ParamTerm.Angle_k: (3, 1),
            ParamTerm.Angle_theta: (3, 1),
            ParamTerm.ProperTorsion_k: (4, PROPERTORSION_TERMS),
            ParamTerm.ImproperTorsion_k: (2, 1),
        }
        self.topo_param_map = TopoParams.topo_param_map.copy()
        self.topo_param_map_rev = TopoParams.topo_param_map_rev.copy()

        if not self.use_improper_torsion:
            self.topo_param_map.pop(TopoTerm.ImproperTorsion)
            self.topo_param_map_rev.pop(ParamTerm.ImproperTorsion_k)
            self.param_shapes.pop(ParamTerm.ImproperTorsion_k)

        self.tanh_output = tanh_output
        self.node_in_dim = node_dim
        self.edge_in_dim = edge_dim

        self.post_mlp: dict[str, MLP] = nn.ModuleDict()
        self.out_mlp: dict[str, MLP] = nn.ModuleDict()
        self._init_mlps(post_mlp_dims, post_mlp_dims_proper, out_mlp_dims, out_mlp_dims_proper, act)

    def _init_mlps(self, post_mlp_dims, post_mlp_dims_proper, out_mlp_dims, out_mlp_dims_proper, act):
        for param_term, shape in self.param_shapes.items():
            pmd = list(post_mlp_dims_proper
                      ) if post_mlp_dims_proper is not None and param_term == ParamTerm.ProperTorsion_k else list(
                          post_mlp_dims)
            omd = list(out_mlp_dims_proper
                      ) if out_mlp_dims_proper is not None and param_term == ParamTerm.ProperTorsion_k else list(
                          out_mlp_dims)

            if param_term not in [ParamTerm.Charge, ParamTerm.Sigma, ParamTerm.Epsilon]:
                self.post_mlp[param_term.name] = MLP(in_channels=self._get_post_in_dim(shape[0]),
                                                     hidden_channels=pmd[0],
                                                     out_channels=pmd[1],
                                                     num_layers=pmd[2],
                                                     norm=None,
                                                     act=act)
            else:
                pmd[1] = self._get_post_in_dim(shape[0])

            self.out_mlp[param_term.name] = MLP(in_channels=pmd[1],
                                                hidden_channels=omd[0],
                                                out_channels=shape[1],
                                                num_layers=omd[1],
                                                norm=None,
                                                act=act)

    def _get_post_in_dim(self, num_node) -> int:
        return num_node * self.node_in_dim

    def _reset_parameter(self):
        for module in self.post_mlp.values():
            module.reset_parameters()
        for module in self.out_mlp.values():
            module.reset_parameters()

    def _prepare_symmetric_input(self, x_h: Tensor, e_h: Tensor, graph: Data) -> dict[TopoTerm, tuple[Tensor]]:
        """symmetric input for post mlp, e.g. ([0, 1], [1, 0]) for bond"""
        x_terms = {}
        for term in self.topo_param_map:
            index = getattr(graph, term.name + "_index")
            xs = []
            for i in range(index.shape[0]):
                xs.append(x_h[index[i].long()])
            if term == TopoTerm.ImproperTorsion:
                x_terms[term] = (
                    torch.concat([xs[0], xs[1]], dim=-1),
                    torch.concat([xs[0], xs[2]], dim=-1),
                    torch.concat([xs[0], xs[3]], dim=-1),
                )
            else:
                x_terms[term] = (torch.concat(xs, dim=-1), torch.concat(xs[::-1], dim=-1))
        return x_terms

    def _output_parameter(self, x_terms, graph: Data):
        '''post_mlp -> symmetry-preserving pooling -> out_mlp'''

        params = {}
        for p_term, t_term in self.topo_param_map_rev.items():
            if t_term is TopoTerm.Atom:
                params[p_term] = self.out_mlp[p_term.name](x_terms[t_term][0])
            else:
                ys = []
                for x in x_terms[t_term]:
                    res = self.post_mlp[p_term.name](x)
                    ys.append(res)
                params[p_term] = self.out_mlp[p_term.name](sum(ys))
        return params

    def forward(self, x_h: Tensor, e_h: Tensor, graph: Data) -> dict[ParamTerm, Tensor]:
        x_terms = self._prepare_symmetric_input(x_h, e_h, graph)
        params = self._output_parameter(x_terms, graph)
        for term in params:
            d = params[term]
            if self.tanh_output > 0.:
                d = self.tanh_output * torch.tanh(d)
            params[term] = d * ParamData.std_mean[term][0] + ParamData.std_mean[term][1]
        return params


class OutputWithBond(OutputLayer):

    def _get_post_in_dim(self, num_node) -> int:
        return num_node * self.node_in_dim + (num_node - 1) * self.edge_in_dim

    def _prepare_symmetric_input(self, x_h: Tensor, e_h: Tensor, graph: Data):
        '''post_mlp -> symmetry-preserving pooling with bond -> out_mlp'''
        x_terms = {}
        for term in self.topo_param_map:
            index = getattr(graph, term.name + "_index")
            if index.shape[0] > 1:
                edge_idx = getattr(graph, term.name + "_edge_idx")  # [nterm, term_shape - 1]
                edge_idx = to_global_edge_idx(edge_idx, getattr(graph, "nBond") * 2, getattr(graph, "n" + term.name))
            xs = []
            for i in range(index.shape[0]):
                xs.append(x_h[index[i].long()])
                if i < index.shape[0] - 1:
                    xs.append(e_h[edge_idx[:, i]])
            if term is TopoTerm.ImproperTorsion:
                x_terms[term] = (
                    torch.concat([xs[0], xs[1], xs[2]], dim=-1),
                    torch.concat([xs[0], xs[3], xs[4]], dim=-1),
                    torch.concat([xs[0], xs[5], xs[6]], dim=-1),
                )
            elif term is TopoTerm.Atom:
                x_terms[term] = tuple(xs)
            else:
                x_terms[term] = (torch.concat(xs, dim=-1), torch.concat(xs[::-1], dim=-1))

        return x_terms


class OutputWithBondBCC(OutputWithBond):

    def _init_mlps(self, post_mlp_dims, post_mlp_dims_proper, out_mlp_dims, out_mlp_dims_proper, act):
        for param_term, shape in self.param_shapes.items():
            pmd = list(post_mlp_dims_proper
                      ) if post_mlp_dims_proper is not None and param_term == ParamTerm.ProperTorsion_k else list(
                          post_mlp_dims)
            omd = list(out_mlp_dims_proper
                      ) if out_mlp_dims_proper is not None and param_term == ParamTerm.ProperTorsion_k else list(
                          out_mlp_dims)

            if param_term not in [ParamTerm.Sigma, ParamTerm.Epsilon]:
                shape_in = 2 if param_term is ParamTerm.Charge else shape[0]
                self.post_mlp[param_term.name] = MLP(in_channels=self._get_post_in_dim(shape_in),
                                                     hidden_channels=pmd[0],
                                                     out_channels=pmd[1],
                                                     num_layers=pmd[2],
                                                     norm=None,
                                                     act=act)
            else:
                pmd[1] = self._get_post_in_dim(shape[0])

            _act = 'tanh' if param_term is ParamTerm.Charge else act
            _bias = False if param_term is ParamTerm.Charge else True
            self.out_mlp[param_term.name] = MLP(in_channels=pmd[1],
                                                hidden_channels=omd[0],
                                                out_channels=shape[1],
                                                num_layers=omd[1],
                                                norm=None,
                                                bias=_bias,
                                                act=_act)

    def _prepare_symmetric_input(self, x_h: Tensor, e_h: Tensor, graph: Data):
        '''post_mlp -> symmetry-preserving pooling with bond -> out_mlp'''
        x_terms = {}
        for term in self.topo_param_map:
            index = getattr(graph, term.name + "_index")
            if index.shape[0] > 1:
                edge_idx = getattr(graph, term.name + "_edge_idx")  # [nterm, term_shape - 1]
                edge_idx = to_global_edge_idx(edge_idx, getattr(graph, "nBond") * 2, getattr(graph, "n" + term.name))
            xs = []
            for i in range(index.shape[0]):
                xs.append(x_h[index[i].long()])
                if i < index.shape[0] - 1:
                    xs.append(e_h[edge_idx[:, i]])

            # used for bcc
            if term is TopoTerm.Bond:
                x_terms['BCC'] = (torch.concat(xs, dim=-1), torch.concat(xs[::-1], dim=-1))

            if term is TopoTerm.ImproperTorsion:
                x_terms[term] = (
                    torch.concat([xs[0], xs[1], xs[2]], dim=-1),
                    torch.concat([xs[0], xs[3], xs[4]], dim=-1),
                    torch.concat([xs[0], xs[5], xs[6]], dim=-1),
                )
            elif term is TopoTerm.Atom:
                x_terms[term] = tuple(xs)
            else:
                x_terms[term] = (torch.concat(xs, dim=-1), torch.concat(xs[::-1], dim=-1))

        return x_terms

    def _output_parameter(self, x_terms, graph: Data):
        '''post_mlp -> symmetry-preserving pooling -> out_mlp'''

        params = {}
        for p_term, t_term in self.topo_param_map_rev.items():

            if p_term is ParamTerm.Charge:
                # bcc charge
                p_term = ParamTerm.Charge
                y0 = self.post_mlp[p_term.name](x_terms['BCC'][0])
                y1 = self.post_mlp[p_term.name](x_terms['BCC'][1])
                bcc = self.out_mlp[p_term.name](y0 - y1).squeeze(-1)
                bcc = torch.tanh(bcc) * 4.

                # aggregate bcc to get charge
                index = graph.Bond_index.T.long()

                # formal charge
                charge = graph.x[:, 2].clone().to(bcc.dtype)
                # average symmetric atoms
                charge = equi_features(charge, graph.equi_index.long())
                # add bcc
                charge.scatter_add_(0, index[:, 0], bcc)
                charge.scatter_add_(0, index[:, 1], -bcc)
                params[p_term] = charge.unsqueeze(-1)

            elif t_term is TopoTerm.Atom:
                params[p_term] = self.out_mlp[p_term.name](x_terms[t_term][0])

            else:
                ys = []
                for x in x_terms[t_term]:
                    res = self.post_mlp[p_term.name](x)
                    ys.append(res)
                params[p_term] = self.out_mlp[p_term.name](sum(ys))
        return params

    def forward(self, x_h: Tensor, e_h: Tensor, graph: Data) -> dict[ParamTerm, Tensor]:
        x_terms = self._prepare_symmetric_input(x_h, e_h, graph)
        params = self._output_parameter(x_terms, graph)
        for term in params:
            d = params[term]
            if self.tanh_output > 0. and term is not ParamTerm.Charge:
                d = self.tanh_output * torch.tanh(params[term])
                d = d * ParamData.std_mean[term][0] + ParamData.std_mean[term][1]
            params[term] = d
        return params
