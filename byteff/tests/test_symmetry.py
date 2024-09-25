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

import pytest
import torch

from byteff.data import preprocess_mol
from byteff.model import MLParams
from byteff.mol import Molecule
from byteff.utils.definitions import ParamTerm

logger = logging.getLogger(__name__)


@pytest.mark.parametrize('feature_config', [{
    'type': 'ExtendedFeature',
    'atom_embedding_dim': 32,
    'tot_chg_embedding_dim': 8,
    'ring_con_embedding_dim': 8,
    'min_ring_size_embedding_dim': 8,
    'bond_ring_embedding_dim': 8,
    'bond_order_embedding_dim': 8,
    'node_mlp_dims': [64, 256, 2],
    'edge_mlp_dims': [64, 256, 2],
    'act': 'gelu'
}])
@pytest.mark.parametrize('gnn_config', [{
    'type': 'EGTLayer',
    'gnn_dims': [256, 256, 4],
    'jk': 'cat',
    'act': 'gelu',
    'heads': 16,
    'at_channels': 16,
    'ffn_dims': [256, 2]
}])
@pytest.mark.parametrize('output_config', [{
    'type': 'OutputWithBondBCC',
    'use_improper_torsion': True,
}])
@pytest.mark.parametrize('mapped_smiles', [
    '[O:1]([H:2])[H:3]',
    '[O:1]=[S:2](=[O:3])([N-:4][S:5](=[O:6])(=[O:7])[C:8]([F:9])([F:10])[F:11])[C:12]([F:13])([F:14])[F:15]'
])
def test_symmetry(feature_config, gnn_config, output_config, mapped_smiles):
    mol = Molecule.from_mapped_smiles(mapped_smiles)
    graph = preprocess_mol(mol)
    model = MLParams(feature_layer=feature_config, gnn_layer=gnn_config, output_layer=output_config)
    node_feature, edge_feature = model.feature_layer(graph)

    equi_node = graph.equi_index.tolist()
    equi_edge = graph.equi_edge_idx.tolist()

    node_dict, edge_dict = {}, {}
    for i, equiv in enumerate(equi_node):
        if equiv not in node_dict:
            node_dict[equiv] = node_feature[i]
        else:
            assert torch.allclose(node_feature[i], node_dict[equiv])
    for i, equiv in enumerate(equi_edge):
        if equiv not in edge_dict:
            edge_dict[equiv] = edge_feature[i]
        else:
            assert torch.allclose(edge_feature[i], edge_dict[equiv])

    x_h, e_h = model.gnn_layer(graph, node_feature, edge_feature)
    node_dict, edge_dict = {}, {}
    for i, equiv in enumerate(equi_node):
        if equiv not in node_dict:
            node_dict[equiv] = x_h[i]
        else:
            assert torch.allclose(x_h[i], node_dict[equiv], atol=1e-6)
    for i, equiv in enumerate(equi_edge):
        if equiv not in edge_dict:
            edge_dict[equiv] = e_h[i]
        else:
            assert torch.allclose(e_h[i], edge_dict[equiv], atol=1e-6)

    pred = model.output_layer(x_h, e_h, graph)
    charge_dict = {}
    for i, equiv in enumerate(equi_node):
        if equiv not in charge_dict:
            charge_dict[equiv] = pred[ParamTerm.Charge][i]
        else:
            assert torch.allclose(pred[ParamTerm.Charge][i], charge_dict[equiv], atol=1e-6)

    # test total charge
    assert (pred[ParamTerm.Charge].sum() - sum(mol.formal_charges)).abs() < 1e-6
