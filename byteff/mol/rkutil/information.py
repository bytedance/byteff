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
from operator import itemgetter
from typing import Dict, Iterable, List, Union

from rdkit import Chem
from rdkit.Chem import AllChem, TorsionFingerprints
from rdkit.Chem.Lipinski import RotatableBondSmarts
from rdkit.Chem.TorsionFingerprints import CalculateTorsionLists

logger = logging.getLogger(__name__)

##########################################
##           read-only information
##########################################


def show_debug_info(rkmol: Chem.Mol):
    rkmol.Debug()
    for idx, rda in enumerate(rkmol.GetAtoms()):
        print('atom', idx, rda.GetPropsAsDict())
    for idx, rdb in enumerate(rkmol.GetBonds()):
        print('bond', idx, rdb.GetPropsAsDict())
    return


def get_mol_formula(rkmol: Chem.Mol) -> str:
    return Chem.rdMolDescriptors.CalcMolFormula(rkmol)


def get_tfd_propers(rkmol: Chem.Mol) -> List:
    assert isinstance(rkmol, Chem.Mol)
    rot_atom_pairs = rkmol.GetSubstructMatches(RotatableBondSmarts)
    torsions, _ = CalculateTorsionLists(rkmol)
    result = []
    for _ in torsions:
        tor = _[0][0]
        if (tor[1], tor[2]) in rot_atom_pairs or (tor[2], tor[1]) in rot_atom_pairs:
            result.append(tor)
    result.sort(key=itemgetter(0, 1, 2, 3))
    return result


def get_sum_absolute_formal_charges(mol: Chem.Mol) -> int:
    '''get total 'absolute' formal charges'''
    count = 0
    for at in mol.GetAtoms():
        count += abs(at.GetFormalCharge())
    return count


def get_nnz_formal_charges(mol: Chem.Mol) -> int:
    '''get number of atoms with non-zero formal charges'''
    count = 0
    for at in mol.GetAtoms():
        count += 1 if at.GetFormalCharge() != 0 else 0
    return count
