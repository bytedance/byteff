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

from itertools import permutations

import torch

from byteff.forcefield import CFF, get_gaff2_tfs, parse_tfs
from byteff.mol import Molecule
from byteff.utils import get_data_file_path
from byteff.utils.definitions import TopoData, TopoTerm

mol = Molecule(get_data_file_path('torsion_example.xyz', 'byteff.tests.testdata'))
topo = parse_tfs(get_gaff2_tfs(mol))
coords = torch.tensor(mol.conformers[0].coords, dtype=torch.float32).unsqueeze(1)


def test_bond():

    term = TopoTerm.Bond
    indices = topo[term].tolist()

    pred = CFF.energy_force(topo, coords, calc_hessian=True, calc_terms={term})
    pred1 = CFF.energy_force(topo, coords, calc_partial_hessian=True, calc_terms={term})

    conf = 0
    full_hessian = pred['hessian'][..., conf]
    partial_hessian = pred1[f'{term.name}_hessian'][:, conf]
    full_hessian1 = torch.zeros_like(full_hessian)

    for i_term, ids in enumerate(indices):
        for i, ai in enumerate(ids):
            for j, aj in enumerate(ids):
                p = partial_hessian[i_term][i * len(indices[0]) + j].reshape(3, 3)
                full_hessian1[ai * 3:(ai + 1) * 3, aj * 3:(aj + 1) * 3] += p

    assert torch.allclose(full_hessian, full_hessian1)


def test_angle():

    term = TopoTerm.Angle
    indices = topo[term].tolist()

    pred = CFF.energy_force(topo, coords, calc_hessian=True, calc_terms={term})
    pred1 = CFF.energy_force(topo, coords, calc_partial_hessian=True, calc_terms={term})

    conf = 0
    full_hessian = pred['hessian'][..., conf]
    partial_hessian = pred1[f'{term.name}_hessian'][:, conf]
    full_hessian1 = torch.zeros_like(full_hessian)

    for i_term, ids in enumerate(indices):
        for i, ai in enumerate(ids):
            for j, aj in enumerate(ids):
                p = partial_hessian[i_term][i * len(indices[0]) + j].reshape(3, 3)
                full_hessian1[ai * 3:(ai + 1) * 3, aj * 3:(aj + 1) * 3] += p

    assert torch.allclose(full_hessian, full_hessian1)


def test_proper():

    term = TopoTerm.ProperTorsion
    indices = topo[term].tolist()

    pred = CFF.energy_force(topo, coords, calc_hessian=True, calc_terms={term})
    pred1 = CFF.energy_force(topo, coords, calc_partial_hessian=True, calc_terms={term})

    conf = 0
    full_hessian = pred['hessian'][..., conf]
    partial_hessian = pred1[f'{term.name}_hessian'][:, conf]
    full_hessian1 = torch.zeros_like(full_hessian)

    for i_term, ids in enumerate(indices):
        for i, ai in enumerate(ids):
            for j, aj in enumerate(ids):
                p = partial_hessian[i_term][i * len(indices[0]) + j].reshape(3, 3)
                full_hessian1[ai * 3:(ai + 1) * 3, aj * 3:(aj + 1) * 3] += p

    assert torch.allclose(full_hessian, full_hessian1)


def test_improper():

    term = TopoTerm.ImproperTorsion
    indices = topo[term].tolist()

    pred = CFF.energy_force(topo, coords, calc_hessian=True, calc_terms={term})
    pred1 = CFF.energy_force(topo, coords, calc_partial_hessian=True, calc_terms={term})

    conf = 0
    full_hessian = pred['hessian'][..., conf]
    partial_hessian = pred1[f'{term.name}_hessian'][:, conf]
    full_hessian1 = torch.zeros_like(full_hessian)

    for i_term, ids in enumerate(indices):
        for i, ai in enumerate(ids):
            for j, aj in enumerate(ids):
                p = partial_hessian[i_term][i * len(indices[0]) + j].reshape(3, 3)
                full_hessian1[ai * 3:(ai + 1) * 3, aj * 3:(aj + 1) * 3] += p

    assert torch.allclose(full_hessian, full_hessian1)


def test_partial_hessian():

    # calc_terms = {TopoTerm.Bond, TopoTerm.Angle, TopoTerm.ProperTorsion, TopoTerm.ImproperTorsion}

    calc_terms = {TopoTerm.ImproperTorsion}

    pred = CFF.energy_force(topo, coords, calc_hessian=True, calc_terms=calc_terms)
    pred1 = CFF.energy_force(topo, coords, calc_partial_hessian=True, calc_terms=calc_terms)

    conf = 0
    full_hessian = pred['hessian'][..., conf].detach()
    partial_hessians = []
    ids_hessian_map = {}
    recs = {}

    print(full_hessian[1 * 3:1 * 3 + 3, 0 * 3:0 * 3 + 3])
    print(pred1['ImproperTorsion_hessian'][0, 0, 4])

    for term in calc_terms:
        width = TopoData.width[term]
        recs[term] = [[[] for _ in range(width)] for _ in range(width)]
        indices = topo[term].tolist()
        print(indices)
        choices = list(permutations(range(width), 2))
        for ids in indices:
            for i, j in choices:
                if abs(i - j) >= 3 and term is TopoTerm.ProperTorsion:
                    continue
                a0, a1 = ids[i], ids[j]
                if (a0, a1) not in ids_hessian_map:
                    ids_hessian_map[(a0, a1)] = len(partial_hessians)
                    partial_hessians.append(full_hessian[a0 * 3:a0 * 3 + 3, a1 * 3:a1 * 3 + 3].reshape(-1))
                recs[term][i][j].append(ids_hessian_map[(a0, a1)])

    partial_hessians = torch.vstack(partial_hessians)
    partial_hessians_pred = torch.zeros_like(partial_hessians)
    for term in calc_terms:
        width = TopoData.width[term]
        ph = pred1[f'{term.name}_hessian'][:, conf]
        for i in range(width):
            for j in range(width):
                if i == j:
                    continue
                if abs(i - j) >= 3 and term is TopoTerm.ProperTorsion:
                    continue
                idx = torch.LongTensor(recs[term][i][j]).unsqueeze(-1).expand(-1, 9)
                partial_hessians_pred.scatter_add_(0, idx, ph[:, i * width + j])

    assert torch.allclose(partial_hessians, partial_hessians_pred, atol=1e-3)
