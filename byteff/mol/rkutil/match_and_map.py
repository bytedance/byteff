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
from typing import Dict, Iterable, List, Set, Tuple, Union
from warnings import warn

from rdkit import Chem
from rdkit.Chem import rdFMCS

from .sanitize import sanitize_rkmol

logger = logging.getLogger(__name__)


def get_smiles(rkmol: Chem.Mol, isomeric: bool = True, kekulize: bool = False) -> str:
    ''' Returns canonical smiles without H atoms.
        Must use after sanitize!
    '''
    _rkmol = Chem.Mol(rkmol)  # make a copy
    clear_atom_map_num(_rkmol)
    # otherwise may throw 'cannot kekulize' error for some molecules
    _rkmol = Chem.RemoveAllHs(_rkmol, sanitize=False)

    return Chem.MolToSmiles(_rkmol,
                            isomericSmiles=isomeric,
                            kekuleSmiles=kekulize,
                            canonical=True,
                            allBondsExplicit=False,
                            allHsExplicit=False,
                            doRandom=False)


def find_mapped_smarts_matches(rkmol: Chem.Mol,
                               mapped_smarts: Union[Chem.Mol, str],
                               *,
                               sanitize: bool = False,
                               aromaticity: str = 'rdkit',
                               use_chirality: bool = False,
                               match_resonance: bool = False,
                               resonance_flag=0) -> Set[Tuple]:
    """Find all sets of atoms in the provided RDKit molecule that match the provided SMARTS string.
    If no mapped number in smarts, return all matched atoms by set of sorted atom index.

    ** adapted from openff.toolkit.utils.rdkit_wrapper.py **
    """

    # Make a copy of the molecule
    # sanitize is required to update the proper information of rkmol
    if sanitize:
        rkmol = sanitize_rkmol(rkmol, aromaticity=aromaticity)

    # Set up query.
    if isinstance(mapped_smarts, str):
        qmol = Chem.MolFromSmarts(mapped_smarts)  # cannot catch the error
        if qmol is None:
            raise ValueError(f'RDKit could not parse the query string "{mapped_smarts}"')
    else:
        assert isinstance(mapped_smarts, Chem.Mol)
        qmol = mapped_smarts

    # Create atom mapping for query molecule
    # this is necessary to get rid of excessive atoms in patterns like "[#1:1]-[#6X3]~[#7,#8,#9,#16,#17,#35]"
    idx_map = dict()
    for atom in qmol.GetAtoms():
        # this checks the map y in pattern [#x:y].
        # y is zero if not defined in pattern, like [#x]
        pattern_index = atom.GetAtomMapNum()
        if pattern_index != 0:
            idx_map[pattern_index - 1] = atom.GetIdx()
    map_list = [idx_map[x] for x in sorted(idx_map)]

    # In high-symmetry cases with medium-sized molecules, it is very easy to end up with a
    # combinatorial explosion in the number of possible matches.
    max_matches = 10000 // qmol.GetNumAtoms()
    if match_resonance:
        warn('resonance match in rdkit is not reliable, this is only for experimental use', UserWarning)
        suppl = Chem.ResonanceMolSupplier(rkmol, resonance_flag)
        full_matches = suppl.GetSubstructMatches(qmol,
                                                 uniquify=False,
                                                 maxMatches=max_matches,
                                                 useChirality=use_chirality)
    else:
        full_matches = rkmol.GetSubstructMatches(qmol,
                                                 uniquify=False,
                                                 maxMatches=max_matches,
                                                 useChirality=use_chirality)

    if map_list:
        matches = [tuple(match[x] for x in map_list) for match in full_matches]
    else:
        matches = [tuple(sorted(set(match))) for match in full_matches]

    return set(matches)


#################
## fmcs
#################

# https://www.rdkit.org/docs/source/rdkit.Chem.rdFMCS.html#rdkit.Chem.rdFMCS.FindMCS
# these two are more important options

# class rdkit.Chem.rdFMCS.AtomCompare
# CompareAny = rdkit.Chem.rdFMCS.AtomCompare.CompareAny
# CompareAnyHeavyAtom = rdkit.Chem.rdFMCS.AtomCompare.CompareAnyHeavyAtom
# CompareElements = rdkit.Chem.rdFMCS.AtomCompare.CompareElements
# CompareIsotopes = rdkit.Chem.rdFMCS.AtomCompare.CompareIsotopes

# class rdkit.Chem.rdFMCS.BondCompare
# CompareAny = rdkit.Chem.rdFMCS.BondCompare.CompareAny
# CompareOrder = rdkit.Chem.rdFMCS.BondCompare.CompareOrder
# CompareOrderExact = rdkit.Chem.rdFMCS.BondCompare.CompareOrderExact


def find_indices_mapping_between_mols(rkmol0: Chem.Mol,
                                      rkmol1: Chem.Mol,
                                      verbose: bool = False,
                                      ring_matches_ring_only: bool = True,
                                      **kwargs) -> Dict[int, int]:
    """return the indices mapping from rkmol1 to rkmol0.
    """
    result = rdFMCS.FindMCS([rkmol0, rkmol1],
                            timeout=5,
                            verbose=verbose,
                            ringMatchesRingOnly=ring_matches_ring_only,
                            **kwargs)
    smarts = result.smartsString
    mcs_mol = Chem.MolFromSmarts(smarts)
    match0 = rkmol0.GetSubstructMatch(mcs_mol)
    match1 = rkmol1.GetSubstructMatch(mcs_mol)
    assert len(match0) == len(match1)
    match_1_to_0 = dict(zip(match1, match0))
    return match_1_to_0


def find_indices_mapping_between_isomorphic_mols(rkmol0: Chem.Mol,
                                                 rkmol1: Chem.Mol,
                                                 verbose: bool = False,
                                                 ring_matches_ring_only: bool = True,
                                                 **kwargs) -> List[int]:
    """return the indices mapping from rkmol1 to rkmol0.
    """
    result = rdFMCS.FindMCS([rkmol0, rkmol1],
                            timeout=5,
                            verbose=verbose,
                            ringMatchesRingOnly=ring_matches_ring_only,
                            **kwargs)
    smarts = result.smartsString
    mcs_mol = Chem.MolFromSmarts(smarts)
    match0 = rkmol0.GetSubstructMatch(mcs_mol)
    match1 = rkmol1.GetSubstructMatch(mcs_mol)
    assert len(match0) == len(match1) == rkmol0.GetNumAtoms()
    match_1_to_0 = dict(zip(match1, match0))
    return [match_1_to_0[i] for i in range(len(match0))]


#################
## atom map num
#################


def is_atom_map_num_valid(rkmol: Chem.Mol) -> bool:
    mapnum = [(i, at.GetAtomMapNum()) for i, at in enumerate(rkmol.GetAtoms())]
    # map number from 1 to N
    return sorted([atom_mapnum - 1 for _, atom_mapnum in mapnum]) == list(range(rkmol.GetNumAtoms()))


def add_atom_map_num(rkmol: Chem.Mol, smarts_atoms: Union[Dict, Iterable[int]] = None) -> None:
    ''' Set atom map number. 
        If smarts_atoms is not given, add number by atom index + 1
    '''
    if smarts_atoms is None:
        for i, at in enumerate(rkmol.GetAtoms()):
            at.SetAtomMapNum(i + 1)
    elif isinstance(smarts_atoms, dict):
        for k, v in smarts_atoms.items():
            rkmol.GetAtomWithIdx(k).SetAtomMapNum(v)
    else:
        for i, k in enumerate(smarts_atoms):
            rkmol.GetAtomWithIdx(k).SetAtomMapNum(i + 1)
    return


def clear_atom_map_num(rkmol: Chem.Mol) -> None:
    for _, at in enumerate(rkmol.GetAtoms()):
        at.SetAtomMapNum(0)
    return


def renumber_atoms_with_atom_map_num(rkmol: Chem.Mol) -> Chem.Mol:
    mapnum = [(i, at.GetAtomMapNum()) for i, at in enumerate(rkmol.GetAtoms())]
    # map number from 1 to N
    # index from 0 to N-1
    assert sorted([atom_mapnum - 1 for _, atom_mapnum in mapnum]) == list(range(rkmol.GetNumAtoms()))
    assert sorted([atom_idx for atom_idx, _ in mapnum]) == list(range(rkmol.GetNumAtoms()))
    mapnum.sort(key=itemgetter(1))
    # rdkit bug: this is a copy but does not copy all conformer props to the returned rkmol
    rkmol = Chem.RenumberAtoms(rkmol, [atom_idx for atom_idx, _ in mapnum])
    return rkmol
