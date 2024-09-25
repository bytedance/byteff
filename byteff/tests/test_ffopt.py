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
from torch_geometric.loader import DataLoader

from byteff.data import LabelType, MolDataset
from byteff.forcefield import CFF
from byteff.forcefield.ff_kernels import dihedral_jacobian
from byteff.forcefield.ffopt import (ConstraintFFopt, _batch_dot, _batch_to_atoms)
from byteff.utils import get_data_file_path, temporary_cd
from byteff.utils.definitions import ParamTerm, TopoParams, TopoTerm

torch.manual_seed(42)
batch_size = 20
nconfs = 25
partial_charges_key = 'am1bcc_charges'
meta_file = get_data_file_path("dataset_torsion/meta.txt", "byteff.tests.testdata")
with temporary_cd():
    dataset = MolDataset(root="./torsion",
                         meta_file=meta_file,
                         partial_charges_key=partial_charges_key,
                         label_type=LabelType.Torsion | LabelType.GAFF2,
                         max_nconfs=nconfs)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)


def test_adjust_grad():

    graph = next(iter(dataloader))
    coords, batch = graph.coords, graph.batch.unsqueeze(-1).expand(-1, nconfs)
    _, jac = dihedral_jacobian(coords, graph.torsion_index)
    grad = torch.rand_like(coords) * 0.1
    new_grad = ConstraintFFopt.adjust_grad(grad, jac, batch)
    assert (_batch_dot(grad, jac, batch).abs() > 1e-5).any()
    assert (_batch_dot(new_grad, jac, batch).abs() * graph.confmask < 1e-5).all()


def test_adjust_position():

    graph = next(iter(dataloader))
    coords, batch = graph.coords, graph.batch.unsqueeze(-1).expand(-1, nconfs)

    def jacobian_func(coords):
        phi, jac = dihedral_jacobian(coords, graph.torsion_index)
        return phi, jac

    threshold = 1e-5
    init_phi, _ = jacobian_func(coords)
    for i in range(10):
        torch.manual_seed(i)
        update = torch.rand_like(coords) * 0.1
        update *= _batch_to_atoms(graph.confmask, batch)
        update, flag = ConstraintFFopt.adjust_position(coords, update, batch, jacobian_func, threshold=threshold)
        assert flag.sum() == nconfs * batch_size
        phi, _ = jacobian_func(coords + update)
        diff = (phi - init_phi + torch.pi) % (2 * torch.pi) - torch.pi
        assert (torch.abs(diff) < threshold).all()

        update = torch.rand_like(coords) * 10.
        update *= _batch_to_atoms(graph.confmask, batch)
        _, flag = ConstraintFFopt.adjust_position(coords, update, batch, jacobian_func, threshold=threshold)
        assert flag.sum() < nconfs * batch_size


def test_optimize():

    graph = next(iter(dataloader))

    topo_params = TopoParams()
    for term in TopoTerm:
        topo_params[term] = getattr(graph, term.name + '_index').T
        topo_params.counts[term] = getattr(graph, 'n' + term.name)
    for term in ParamTerm:
        topo_params[term] = getattr(graph, term.name + '_label')

    torsion_index = graph.torsion_index

    def energy_func(coords):
        graph.coords = coords
        pred = CFF.energy_force(topo_params, graph.coords, confmask=graph.confmask)
        # print(pred['energy'][:5, 0])
        return pred['energy'], pred['forces']

    def jacobian_func(coords):
        phi, jac = dihedral_jacobian(coords, torsion_index)
        return phi, jac

    old_energy, _ = energy_func(graph.coords)

    new_coords, converge_flag = ConstraintFFopt.optimize(graph,
                                                         energy_func=energy_func,
                                                         jacobian_func=jacobian_func,
                                                         pos_res_k=10.,
                                                         max_iter=300,
                                                         max_step=0.1,
                                                         f_max=0.01)

    new_energy, _ = energy_func(new_coords)

    assert (new_energy <= old_energy).all()
    assert converge_flag.all()
