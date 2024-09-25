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

from collections import defaultdict

from rdkit import Chem


def find_symmetry_rank(rkmol: Chem.Mol) -> list[int]:
    """Return a canonical rank. 
       Symmetric atoms are assigned the same rank.
       Fails in special cases, i.g. 'C1CN(CCOCCN2CCOCC2)CCO1'.
    """
    rkmol = Chem.Mol(rkmol)

    # modify atom/bond feature
    for bond in rkmol.GetBonds():
        bond.SetBondType(Chem.rdchem.BondType.SINGLE)
        bond.SetIsAromatic(False)
        bond.SetIsConjugated(False)
    for atom in rkmol.GetAtoms():
        atom.SetFormalCharge(0)
        atom.SetIsAromatic(False)
        atom.SetIsotope(0)
        atom.SetHybridization(Chem.HybridizationType.S)

    canon_rank = list(Chem.CanonicalRankAtoms(rkmol, breakTies=False, includeChirality=False))
    return canon_rank


def find_equivalent_atoms(rkmol: Chem.Mol) -> dict[int, list[int]]:

    canon_rank = find_symmetry_rank(rkmol)
    book = defaultdict(list)
    for i, rank in enumerate(canon_rank):
        book[rank].append(i)
    equiv_record = defaultdict(list)
    for lst in book.values():
        if len(lst) > 1:
            lst.sort()
            equiv_record[lst[0]] = lst
    return dict(equiv_record)
