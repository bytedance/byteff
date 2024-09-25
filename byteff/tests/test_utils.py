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

from byteff.mol import Molecule, moltools, rkutil


def test_find_equivalent_bonds_angles():

    mol = Molecule.from_smiles('CC([O-])=O')

    equi_bonds, equi_angles = moltools.find_equivalent_bonds_angles(mol)

    assert len(equi_bonds) == 2
    assert {(1, 2), (1, 3)} in equi_bonds

    assert len(equi_angles) == 3
    assert {(0, 1, 2), (0, 1, 3)} in equi_angles


def test_find_equivalent_index():

    mol = Molecule.from_smiles('[O-]S(=O)(=O)CC(C)C([O-])=O')
    bonds = mol.get_bonds()

    equi_atom_idx, equi_bond_idx = moltools.find_equivalent_index(mol, bonds)

    atom_rank = rkutil.find_symmetry_rank(mol.get_rkmol())

    for i in range(mol.natoms - 1):
        for j in range(i, mol.natoms):
            if equi_atom_idx[i] == equi_atom_idx[j]:
                assert atom_rank[i] == atom_rank[j]
            else:
                assert atom_rank[i] != atom_rank[j]

    for i in range(len(bonds) - 1):
        for j in range(i, len(bonds)):
            bond_rank_i = sorted([atom_rank[b] for b in bonds[i]])
            bond_rank_j = sorted([atom_rank[b] for b in bonds[j]])
            if equi_bond_idx[i] == equi_bond_idx[j]:
                assert bond_rank_i == bond_rank_j
            else:
                assert bond_rank_i != bond_rank_j
