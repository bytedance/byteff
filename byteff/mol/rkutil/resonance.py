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

import copy
import logging
from collections import defaultdict
from operator import itemgetter
from typing import List

from rdkit import Chem

from .information import (get_nnz_formal_charges, get_sum_absolute_formal_charges)
from .match_and_map import get_smiles
from .sanitize import sanitize_rkmol

logger = logging.getLogger(__name__)


def get_resonance_structures(mol: Chem.Mol, flags: int = 0, filter_by_formal_charges: bool = True) -> List[Chem.Mol]:
    # doc:
    # https://www.rdkit.org/docs/cppapi/classRDKit_1_1ResonanceMolSupplier.html
    # https://www.rdkit.org/docs/source/rdkit.Chem.rdchem.html?highlight=resonance#rdkit.Chem.rdchem.ResonanceMolSupplier
    # flag numbers:
    # 1: rdkit.Chem.rdchem.ResonanceFlags.ALLOW_INCOMPLETE_OCTETS
    # 2: rdkit.Chem.rdchem.ResonanceFlags.ALLOW_CHARGE_SEPARATION
    # 4: rdkit.Chem.rdchem.ResonanceFlags.KEKULE_ALL
    # 8: rdkit.Chem.rdchem.ResonanceFlags.UNCONSTRAINED_CATIONS
    # 16: rdkit.Chem.rdchem.ResonanceFlags.UNCONSTRAINED_ANIONS
    # flag = 0 means these flags are all turned off. This behavior is usally correct for forcefield development.

    suppl = Chem.ResonanceMolSupplier(mol, flags)
    suppl.SetNumThreads(1)
    suppl.SetProgressCallback(None)

    # rdkit bug: this works around some freezing cases, but may loose many resoners in some other cases.

    # workaround in https://github.com/rdkit/rdkit/issues/6704
    # from rdkit.Chem.rdchem import ResonanceMolSupplierCallback
    # class EmptyResonanceMolSupplierCallback(ResonanceMolSupplierCallback):

    #     def __call__(self):
    #         pass

    # suppl.SetProgressCallback(EmptyResonanceMolSupplierCallback())

    resMols = list(suppl)

    # prefilter by total formal charges
    # because rdkit may have bug
    origin_net_charge = Chem.GetFormalCharge(mol)
    prefiltered = []
    for m in resMols:
        if Chem.GetFormalCharge(m) == origin_net_charge:
            prefiltered.append(m)

    resMols = prefiltered

    if len(resMols) == 0:
        logger.warning('No resoner found for this molecule, returning itself')
        return [mol]

    if not filter_by_formal_charges:
        return resMols

    # first filter by number of atoms
    filtered = defaultdict(list)
    for mol in resMols:
        count = get_nnz_formal_charges(mol)
        filtered[count].append(mol)

    # then filter by total formal charges
    min_count = min(filtered.keys())
    filtered2 = defaultdict(list)
    for mol in filtered[min_count]:
        count = get_sum_absolute_formal_charges(mol)
        filtered2[count].append(mol)
    return filtered2[min(filtered2.keys())]


def get_canonical_resoner(input_mol: Chem.Mol) -> Chem.Mol:
    '''not really a true 'canonical'. make a reasonable best guess in most situations. 
    it is recommended that mol has no stereochemistry info 

    two cases: 
    1. mol is already a reasonable structure
    2. mol is not a reasonable structure
    '''

    origin_smiles = get_smiles(input_mol)

    cdd_mols = get_resonance_structures(input_mol, flags=0, filter_by_formal_charges=True)

    # find the one with the smallest number for formal charges
    # print('---------------')
    # print(Chem.MolToSmiles(target), target_formal_charges)
    smiles_mol_map = {}
    for rkmol in cdd_mols:
        _rkmol = sanitize_rkmol(rkmol)
        smi = get_smiles(_rkmol)
        smiles_mol_map[smi] = _rkmol

    # logger.info('%s', smiles_mol_map)

    if origin_smiles in smiles_mol_map:
        return copy.deepcopy(input_mol)
    else:
        # return the mol corresponding to the 'smallest' smiles string
        return copy.deepcopy(sorted(list((smi, mol) for smi, mol in smiles_mol_map.items()), key=itemgetter(0))[0][1])
