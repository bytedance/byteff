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
from typing import Dict, Iterable, Tuple

from rdkit import Chem
from rdkit.Chem import AllChem, Draw, rdDepictor

logger = logging.getLogger(__name__)

# https://www.rdkit.org/docs/RDKit_Book.html#feature-flags-global-variables-affecting-rdkit-behavior
# preferCoordGen:
# when this is true Schrodinger’s open-source Coordgen library will be used to generate 2D coordinates of molecules.
# The default value is false. This can be set from C++ using the variable RDKit::RDDepict::preferCoordGen or
# from Python using the function rdDepictor.SetPreferCoordGen(). Added in the 2018.03 release.
rdDepictor.SetPreferCoordGen(True)

##########################################
##           read-only plot
##########################################

DEFAULT_PLOT_SHAPE = (600, 600)


def _remove_implicit_hs(mol: Chem.Mol) -> Tuple[Chem.Mol, Dict]:
    _mol = Chem.Mol(mol)
    for atom in _mol.GetAtoms():
        atom.SetIntProp("OldIndex", atom.GetIdx())
    _mol = Chem.RemoveHs(_mol)
    id_map = dict()
    for atom in _mol.GetAtoms():
        id_map[atom.GetIntProp("OldIndex")] = atom.GetIdx()
    return _mol, id_map


def plot_molecule_torsion(rkmol: Chem.Mol, torsion_list: Iterable[Iterable], molecule_name: str = 'molecule'):
    assert len(torsion_list[0]) == 4
    tempfilename = '{}.sdf'.format(molecule_name)
    writer = Chem.SDWriter(tempfilename)
    writer.write(rkmol, confId=0)

    # construct indices map from with-hygrogens to no-hydrogens
    mol_hs = Chem.MolFromMolFile(tempfilename, removeHs=False)
    no_hs_indices_map = dict()
    no_hs_index = 0
    for atom in mol_hs.GetAtoms():
        if atom.GetSymbol() != 'H':
            no_hs_indices_map[atom.GetIdx()] = no_hs_index
            no_hs_index += 1
    # save one figure for each torsion group
    for torsion_atom_indices in torsion_list:
        no_hs_torsion_atom_indices = tuple((no_hs_indices_map[idx] for idx in torsion_atom_indices))
        # make a copy of rkmol by reading a file
        mol = Chem.MolFromMolFile(tempfilename)
        Chem.rdDepictor.Compute2DCoords(mol)
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(atom.GetIdx())
        img = Draw.MolToImage(mol, size=DEFAULT_PLOT_SHAPE, highlightAtoms=no_hs_torsion_atom_indices)
        img.save('{}_torsion_{}_structure.jpg'.format(molecule_name, '_'.join([str(i) for i in torsion_atom_indices])))
    return


def show_mol(mol: Chem.Mol,
             size: Tuple[int, int] = DEFAULT_PLOT_SHAPE,
             *,
             highlight: Iterable = None,
             remove_h: bool = False,
             plot_kekulize: bool = False,
             idx_base_1: bool = False):
    '''draw molecule (copy) with highlight'''

    _mol = Chem.Mol(mol)

    _highlight = []
    base = 1 if idx_base_1 else 0
    for atom in _mol.GetAtoms():
        atom.SetProp('atomNote', str(atom.GetIdx() + base))

    if remove_h:
        _mol, id_map = _remove_implicit_hs(_mol)
        if highlight is not None:
            for k in highlight:
                if k in id_map:
                    _highlight.append(id_map[k])
    else:
        if highlight is not None:
            _highlight = list(highlight)

    AllChem.Compute2DCoords(_mol)
    highlight_bond = []
    if _highlight:
        for i in range(len(_highlight) - 1):
            for j in range(i + 1, len(_highlight)):
                bond = _mol.GetBondBetweenAtoms(_highlight[i], _highlight[j])
                if bond is not None:
                    highlight_bond.append(bond.GetIdx())

    img = Draw.MolToImage(_mol,
                          size=size,
                          kekulize=plot_kekulize,
                          fitImage=True,
                          highlightAtoms=_highlight,
                          highlightBonds=highlight_bond)

    return img


def show_mol_grid(mols: Iterable[Chem.Mol],
                  size: Tuple[int, int] = DEFAULT_PLOT_SHAPE,
                  mol_per_row: int = 2,
                  *,
                  highlights: Iterable[Iterable] = None,
                  legends: Iterable = None,
                  remove_h: bool = False):
    '''draw molecule grid with highlight'''
    new_mols = []
    new_highlights = []
    for i, mol in enumerate(mols):
        if remove_h:
            _mol, id_map = _remove_implicit_hs(mol)
            if highlights is not None:
                _highlight = []
                for k in highlights[i]:
                    if k in id_map:
                        _highlight.append(id_map[k])
            else:
                _highlight = []
        else:
            _mol = mol
            _highlight = [] if highlights is None else list(highlights[i]).copy()
        new_mols.append(_mol)
        new_highlights.append(_highlight)

    img = Draw.MolsToGridImage(new_mols,
                               molsPerRow=mol_per_row,
                               subImgSize=size,
                               legends=legends,
                               highlightAtomLists=new_highlights,
                               useSVG=False,
                               returnPNG=False)
    return img


def show_smarts(smarts: str, size: Tuple[int, int] = DEFAULT_PLOT_SHAPE, kekulize: bool = False):
    '''draw mol pattern given smarts'''
    mol = Chem.MolFromSmarts(smarts)
    AllChem.Compute2DCoords(mol)
    return Draw.MolToImage(mol, size=size, kekulize=kekulize, fitImage=True)


# these are currently not used by any function
# def show_bond_label(mol: Chem.Mol, label_dict: Dict, label: str = "bondNote", size=(800, 800)):
#     '''draw molecule with bond label'''
#     _mol = Chem.Mol(mol)
#     AllChem.Compute2DCoords(_mol)
#     for bond in _mol.GetBonds():
#         bond_t = sorted_atomids((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
#         bond.SetProp(label, label_dict[bond_t])
#     return Draw.MolToImage(_mol, size=size, kekulize=False, fitImage=True)

# def show_mol(mol: Chem.Mol, label: str = "atomNote", size: Tuple[int, int] = (800, 800)):
#     '''draw molecule with atom number'''
#     _mol = Chem.Mol(mol)
#     AllChem.Compute2DCoords(_mol)
#     for atom in _mol.GetAtoms():
#         atom.SetProp(label, str(atom.GetIdx()))
#     return Draw.MolToImage(_mol, size=size, kekulize=False, fitImage=True)
