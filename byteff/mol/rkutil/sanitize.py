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
import traceback

import rdkit
from packaging import version
from rdkit import Chem
from rdkit.Chem import rdChemReactions, rdCIPLabeler

from .information import show_debug_info

logger = logging.getLogger(__name__)

##########################################
##       molecule process pipeline
##########################################

# check rdkit version, allow 2023.3 and 2023.9
assert version.parse(rdkit.rdBase.rdkitVersion).major == 2023, version.parse(rdkit.rdBase.rdkitVersion)

# these are added in rdkit 2022.9
# for 2022.3 these do not exist
Chem.SetAllowNontetrahedralChirality(False)
Chem.SetUseLegacyStereoPerception(False)


def get_mol_from_smiles(smiles: str, *, debug: bool = False, **kwargs) -> Chem.Mol:

    if 'sanitize' in kwargs:
        logger.warning('sanitize option currently has no effect in this function')

    # https://www.rdkit.org/docs/source/rdkit.Chem.rdmolfiles.html#rdkit.Chem.rdmolfiles.SmilesParserParams
    # property allowCXSMILES
    # controls whether or not the CXSMILES extensions are parsed
    # property debugParse
    # controls the amount of debugging information produced
    # property parseName
    # controls whether or not the molecule name is also parsed
    # property removeHs
    # controls whether or not Hs are removed before the molecule is returned
    # property sanitize
    # controls whether or not the molecule is sanitized before being returned
    # property strictCXSMILES
    # controls whether or not problems in CXSMILES parsing causes molecule parsing to fail

    sp = Chem.SmilesParserParams()
    sp.allowCXSMILES = False
    sp.debugParse = debug
    sp.parseName = False
    sp.removeHs = False

    # for rdkit<=2022.3, sanitize=True and useLegacyStereo=True are required to parse stereo correctly
    # sp.sanitize = sanitize  # this uses SANITIZE_ALL flag internally
    # this is removed in 2022.9 and controlled by global switch
    # sp.useLegacyStereo = True  # stick to legacy mode until rdkit new stereo code become stable

    rkmol = Chem.MolFromSmiles(smiles, sp)
    # rkmol.UpdatePropertyCache(strict=False)

    return rkmol


def apply_inplace_reaction(rkmol: Chem.Mol, reaction_transform: str) -> int:
    # demo:
    # https://greglandrum.github.io/rdkit-blog/posts/2021-12-15-single-molecule-reactions.html
    # explanation:
    # https://github.com/rdkit/rdkit/pull/4511

    # RunReactantInPlace
    # 1. preserves AtomProps before/after reaction
    # 2. atomidx mapping before/after is defined by reaction_transform

    transform = rdChemReactions.ReactionFromSmarts(reaction_transform)
    assert transform is not None, f'{reaction_transform} is not valid'
    count = 0
    while transform.RunReactantInPlace(rkmol):
        # one reaction at a time
        # keep reaction until all reactant has been transformed
        count += 1

    return count


# derived from the MolVS set, with ChEMBL-specific additions
# note: some reactant cannot pass the sanitization stage so they are annotated out
normalization_transforms = {
    # name: reaction
    'Nitro to N+(O-)=O': '[N;X3:1](=[O:2])=[O:3]>>[N+1:1]([O-1:2])=[O:3]',
    # these 4-valence N cannot be sanitized before reaction
    # 'Diazonium': '[*:1]-[N;X2:2]#[N;X1:3]>>[*:1]-[*+1:2]#[*:3]',
    # 'Quaternary N': '[N;X4;v4;+0:1]>>[*+1:1]',
    # this 3-valence N cannot be sanitized before reaction
    # 'Trivalent O': '[*:1]=[O;X2;v3;+0:2]-[#6:3]>>[*:1]=[*+1:2]-[*:3]',
    'Sulfoxide to -S+(O-)': '[!O:1][S+0;D3:2](=[O:3])[!O:4]>>[*:1][S+1:2]([O-:3])[*:4]',
    'Sulfoxide to -S+(O-) 2': '[!O:1][SH1+1;D3:2](=[O:3])[!O:4]>>[*:1][S+1:2]([O-:3])[*:4]',
    'Trivalent S': '[O:1]=[S;D2;+0:2]-[#6:3]>>[O:1]=[S+1:2]-[#6:3]',
    'Deprotonated sulfonamide': '[N:1]=[SX4:2]([O-:3])=[O:4]>>[N-:1]-[SX4:2](=[O-0:3])=[O:4]',
    # We do not modify connectivity graph of a molecule, so tautomers are not modified
    # 'Bad amide tautomer1': '[C:1]([OH1;D1:2])=;!@[NH1:3]>>[C:1](=[OH0:2])-[NH2:3]',
    # 'Bad amide tautomer2': '[C:1]([OH1;D1:2])=;!@[NH0:3]>>[C:1](=[OH0:2])-[NH1:3]',
    # we do not need this
    # 'Halogen with no neighbors': '[F,Cl,Br,I;X0;+0:1]>>[*-1:1]',
    # this is separate into two reactions to ensure elements do not change
    # 'Odd pyridine/pyridazine oxide structure': '[C,N;-;D2,D3:1]-[N+2;D3:2]-[O-;D1:3]>>[*;-0:1]=[N+1:2]-[O-:3]',
    'Odd pyridine/pyridazine oxide structure C': '[C;-;D2,D3:1]-[N+2;D3:2]-[O-;D1:3]>>[C;-0:1]=[N+1:2]-[O-:3]',
    'Odd pyridine/pyridazine oxide structure N': '[N;-;D2,D3:1]-[N+2;D3:2]-[O-;D1:3]>>[N;-0:1]=[N+1:2]-[O-:3]',
    'Odd azide': '[*:1][N-:2][N+:3]#[N:4]>>[*:1][N+0:2]=[N+:3]=[N-:4]'
}


def normalize_rkmol(rkmol: Chem.Mol) -> Chem.Mol:
    '''adapted from chembl_structure_pipeline::standardizer
    this should be called after the molecule has been sanitized

    Note: in chembl pipeline, this normalization stage happens before 'sanitization'.
    so some of the rules are necessary, for example, the 'Nitro to N+(O-)=O' rule

    In our pipeline, -N(=O)=O will be rejected because of sanitization error during initialization, either from sdf or from smiles.
    So only a small part of these chembl rules are actually effective in our Molecule class.
    These are kept the same as the chembl original definition to simplify maintenance
    '''

    original_num_atoms = rkmol.GetNumAtoms()

    for name, reaction in normalization_transforms.items():
        count = apply_inplace_reaction(rkmol, reaction)
        if count > 0:
            logger.info(f'{name} transform {reaction} has been applied {count} times.')

    res = rkmol

    assert original_num_atoms == res.GetNumAtoms()

    return res


def sanitize_rkmol(rkmol: Chem.Mol,
                   *,
                   aromaticity: str = 'rdkit',
                   debug: bool = False,
                   allow_advanced_skip_kekulize: bool = False) -> Chem.Mol:
    '''use this sanitize rule everywhere in this project
        returns a sanitized copy
    '''
    _rkmol = Chem.Mol(rkmol)  # deep copy
    # sometimes this input _rkmol is not sanitized yet and rdkit crashes at this step
    # this step has to be commented out
    # input_smiles = Chem.MolToSmiles(_rkmol)

    # stage 1, sanitization
    # sanitize according to default sanitize policy
    # it is important to use ^ Chem.SANITIZE_SETAROMATICITY to clean up aromaticity info already included in rkmol
    # without Chem.SANITIZE_CLEANUPCHIRALITY flag, this removes non-sp3 chirality
    sanitize_flag = Chem.SANITIZE_ALL ^ Chem.SANITIZE_SETAROMATICITY
    try:
        Chem.SanitizeMol(_rkmol, sanitize_flag)
    except Chem.rdchem.KekulizeException as e:
        traceback.print_exc()
        if allow_advanced_skip_kekulize:
            logger.warning('fail to sanitize, retry sanitizing without kekulizing')
            Chem.SanitizeMol(_rkmol, sanitize_flag ^ Chem.SANITIZE_KEKULIZE)
        else:
            raise ValueError('RDKit could not sanitize the molecule') from e
    except ValueError as e:
        traceback.print_exc()
        raise ValueError('RDKit could not sanitize the molecule') from e

    if debug:
        logger.info('_rkmol after sanitization')
        show_debug_info(_rkmol)

    # stage 2, set aromaticity model
    try:
        # rdkit supports several different aromaticity models: simple, rdkit, and MDL
        # https://www.rdkit.org/docs/RDKit_Book.html#aromaticity
        if aromaticity.lower() == 'mdl':
            Chem.SetAromaticity(_rkmol, Chem.AromaticityModel.AROMATICITY_MDL)
        elif aromaticity.lower() == 'rdkit':
            Chem.SetAromaticity(_rkmol, Chem.AromaticityModel.AROMATICITY_RDKIT)
        elif aromaticity.lower() == 'simple':
            Chem.SetAromaticity(_rkmol, Chem.AromaticityModel.AROMATICITY_SIMPLE)
        else:
            raise ValueError(f'unsupported aromaticity {aromaticity}, must be mdl, rdkit, or simple.')
    except ValueError as e:
        traceback.print_exc()
        raise ValueError('RDKit could not set aromaticity') from e

    if debug:
        logger.info('_rkmol after set aromaticity')
        show_debug_info(_rkmol)

    return _rkmol


def cleanup_rkmol_stereochemistry(rkmol: Chem.Mol, *, verbose: bool = False) -> Chem.Mol:
    '''input rkmol must have been sanitized
    return a copy
    '''

    _rkmol = Chem.Mol(rkmol)  # deep copy

    if verbose:
        show_debug_info(_rkmol)

    # internal behavior is controlled by 'Chem.SetUseLegacyStereoPerception(False)'
    if _rkmol.GetNumConformers() > 0:
        # if stereo is important to you, we assume all conformers have the same stereochemistry
        # do not remove this line, even with rdkit >= 2022.9
        Chem.AssignStereochemistryFrom3D(_rkmol, confId=0, replaceExistingTags=True)
    else:
        Chem.SetBondStereoFromDirections(_rkmol)
        Chem.AssignStereochemistry(_rkmol, cleanIt=True, force=True, flagPossibleStereoCenters=True)

    rdCIPLabeler.AssignCIPLabels(_rkmol)

    if verbose:
        show_debug_info(_rkmol)

    return _rkmol


def cleanup_rkmol_isotope(rkmol: Chem.Mol):
    '''input rkmol must have been sanitized
    return a copy
    '''
    _rkmol = Chem.Mol(rkmol)  # deep copy
    atom_data = [(atom, atom.GetIsotope()) for atom in _rkmol.GetAtoms()]
    for atom, isotope in atom_data:
        if isotope:
            atom.SetIsotope(0)
    return _rkmol
