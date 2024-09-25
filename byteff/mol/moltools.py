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

import io
from collections import defaultdict
from typing import Iterator, Sequence, Union

import numpy as np

from byteff.mol import Molecule, MoleculeGraph, rkutil
from byteff.mol.moleculegraph import Hybridization
from byteff.utils.tar import MultiTarFileWithMeta


def match_linear_proper(mol: Molecule):
    linear_patterns = ["[*:1]~[#6X2;!r5;!r6:2]~[*:3]~[*:4]", "[!$([#7H1]):1]~[#7X2+:2]~[!$([#7H1]):3]~[*:4]"]
    rkmol = mol.get_rkmol()
    match_results = set()
    for pattern in linear_patterns:
        matches = rkutil.find_mapped_smarts_matches(rkmol, pattern)
        for atomidxs in matches:
            ordered_atomidxs = rkutil.sorted_atomids(atomidxs)
            match_results.add(ordered_atomidxs)  # update, overwrite previous match
    return match_results


def match_methyl_proper(mol: Molecule):
    methyl_pattern = "[#1:1]~[#6X4H3:2]~[*:3]~[*:4]"
    rkmol = mol.get_rkmol()
    match_results = set()
    matches = rkutil.find_mapped_smarts_matches(rkmol, methyl_pattern)
    for atomidxs in matches:
        ordered_atomidxs = rkutil.sorted_atomids(atomidxs)
        match_results.add(ordered_atomidxs)  # update, overwrite previous match
    return match_results


def judge_mol_trainable(mol: Molecule):

    graph = MoleculeGraph(mol)
    for atom in graph.get_atoms():
        if atom.hybrdization not in {Hybridization.S, Hybridization.SP, Hybridization.SP2, Hybridization.SP3}:
            return False
        if atom.atomic_number in {17, 35, 53} and atom.connectivity > 1:
            return False
    return True


def find_equivalent_bonds_angles(mol: Molecule) -> tuple[list[set[tuple]], list[set[tuple]]]:
    atom_ranks = rkutil.find_symmetry_rank(mol.get_rkmol())
    graph = MoleculeGraph(mol)

    bond_rec = defaultdict(set)
    for bond in graph.get_bonds():
        bond_rec[rkutil.sorted_atomids([atom_ranks[bond[t]] for t in range(2)])].add(bond)
    equi_bonds = [v for v in bond_rec.values() if len(v) > 1]

    angle_rec = defaultdict(set)
    for angle in graph.get_angles():
        angle_rec[rkutil.sorted_atomids([atom_ranks[angle[t]] for t in range(3)])].add(angle)
    equi_angles = [v for v in angle_rec.values() if len(v) > 1]
    return equi_bonds, equi_angles


def find_equivalent_index(mol: Molecule, bond_index: list[tuple[int]]) -> tuple[list[int], list[int]]:
    atom_ranks = rkutil.find_symmetry_rank(mol.get_rkmol())
    atom_rec = {}
    atom_equi_index = []
    for i, rank in enumerate(atom_ranks):
        if rank in atom_rec:
            atom_equi_index.append(atom_rec[rank])
        else:
            atom_rec[rank] = i
            atom_equi_index.append(i)

    bond_rec = {}
    bond_equi_index = []
    for i, bond in enumerate(bond_index):
        bond_rank = tuple(sorted([atom_ranks[b] for b in bond]))
        if bond_rank in bond_rec:
            bond_equi_index.append(bond_rec[bond_rank])
        else:
            bond_rec[bond_rank] = i
            bond_equi_index.append(i)
    return atom_equi_index, bond_equi_index


class MolTarLoader(MultiTarFileWithMeta):
    """
    Load Molecule from meta file and tar files.
    The directory should look like this:
        hessian/
            ├── meta.txt
            └── tar_files
                ├── 01.tar
                ├── ...
                └── 24.tar
    The meta file should look like this:
        01.tar,01/1000040.xyz,01/1000040.npz
        01.tar,01/1000043.xyz,01/1000043.npz
        01.tar,01/1000044.xyz,01/1000044.npz
    """

    def __init__(self,
                 data_dir: str,
                 meta_file="meta.txt",
                 tar_dir="tar_files",
                 name_filter: Sequence[str] = None) -> None:
        super().__init__(data_dir, meta_file=meta_file, tar_dir=tar_dir, name_filter=name_filter)

    def _post_process(self, data: dict[str, bytes]) -> Molecule:
        if data is None:
            return None
        xyz_name, npz_name = "", ""
        for k in data.keys():
            if k.endswith(".xyz"):
                xyz_name = k
            elif k.endswith(".npz"):
                npz_name = k
        mol = Molecule.from_xyz(io.StringIO(data[xyz_name].decode()))
        mol.name = xyz_name[:-4]
        if npz_name:
            npz = data[npz_name]
            npzdata = np.load(io.BytesIO(npz))
            mol.conformers[0].confdata['hessian'] = npzdata['hessian']
        return mol

    def __getitem__(self, item: Union[str, int, slice]) -> Union[Molecule, Iterator[Molecule]]:
        return super().__getitem__(item)
