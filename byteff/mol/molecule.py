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
import hashlib
import io
import logging
import re
import traceback
from ast import literal_eval
from collections import Counter, OrderedDict, defaultdict
from operator import itemgetter
from typing import Dict, Iterable, List, Tuple, Union

import ase.io as aseio
import numpy as np
import rdkit.Chem as Chem
from PIL.Image import Image
from rdkit.Chem.Descriptors import NumRadicalElectrons

from byteff.mol import rkutil
from byteff.mol.conformer import (Conformer, is_charge_key, is_energy_key, is_force_key, is_smiles_key,
                                  is_topo_params_key, prefix_prop_key, write_conformers_to_extxyz,
                                  xyz_properties_parser)
from byteff.utils.definitions import TopoParams

logger = logging.getLogger(__name__)

################################################################################
# the core molecule class to unify the topology, molecule data, and atom data
# hanles only REAL molecules, NOT for FEP alchemical molecules
# Length UNIT in A, same as rdkit and ase, to simplify development
# Energy/Force UNIT is managed by user in the moledata/confdata fields
################################################################################

##########################################
# stereo:
# 1. if the input contains or generates at least one conformer, stereochemistry is assigned according to conformer 0
# 2. if the input contains no conformer (only smiles/mapped smiles), stereochemistry is assigned according to input
##########################################


class MappedSmilesMissingException(Exception):
    pass


class Molecule:

    def __init__(self,
                 molecule: Union[str, Chem.Mol],
                 fmt: str = '',
                 name: str = '',
                 keep_conformers: bool = True,
                 keep_mol_prop: bool = False,
                 **kwargs):
        self.rkmol = None
        self._name = ''
        self.name = name
        self.moledata = dict()
        self._conformers = []

        # intrinsic moledata as properties
        # these do not exist in self.moledata as keys
        self._natoms = None
        self._nconfs = None
        self._atomic_symbols = None
        self._atomic_numbers = None
        self._formal_charges = None
        self._atomic_masses = None

        self._aromaticity = kwargs.get('aromaticity', 'rdkit')  # default to rdkit, unless you manually specify mdl
        assert self._aromaticity in ['rdkit',
                                     'mdl'], f'aromaticity must be either rdkit or mdl, got {self._aromaticity}'

        if isinstance(molecule, Chem.Mol):
            # number of conformers = number of conformers included in molecule
            self._from_rkmol(molecule, keep_conformers=keep_conformers, keep_mol_prop=keep_mol_prop, **kwargs)
        else:
            if isinstance(molecule, str):
                # infer fmt from ext
                supported_exts = {"sdf", "xyz"}
                ext = molecule.split(".")[-1]
                ext = "xyz" if ext == "extxyz" else ext  # add support for extxyz
                if ext in supported_exts:
                    if fmt:
                        assert fmt == ext, f"assigned format {fmt} != file ext .{ext}"
                    else:
                        fmt = ext
            assert fmt in {"sdf", "xyz", "smiles", "mapped_smiles"}
            if fmt == 'xyz':
                assert isinstance(molecule, (io.StringIO, str))
                self._from_xyz(molecule, keep_conformers=keep_conformers, **kwargs)
            else:
                assert isinstance(molecule, str)
                if fmt == 'sdf':
                    kwargs['keep_mol_prop'] = keep_mol_prop
                    self._from_sdf(molecule, keep_conformers=keep_conformers, **kwargs)

                elif fmt == 'smiles':
                    self._from_smiles(molecule, kwargs.pop('nconfs', 0), **kwargs)
                elif fmt == "mapped_smiles":
                    self._from_mapped_smiles(molecule, kwargs.pop('nconfs', 0), **kwargs)

        self._finish(**kwargs)

        return

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value: str):
        assert isinstance(value, (np.integer, np.floating, str))
        value = str(value)
        # formula may include + and - for total charge
        # allow only alphanumeric characters, -, + and _
        # allow / for hierarchy structure
        assert bool(re.match('^[a-zA-Z0-9_+-/]*$', value)), value
        self._name = value

    def _normalize(self):
        self.rkmol = rkutil.normalize_rkmol(self.rkmol)
        return

    def _sanitize(self):
        '''the default aromaticity model is 'rdkit'. If you need 'mdl', call sanitize(aromaticity='mdl')'''
        rkmol = rkutil.sanitize_rkmol(self.rkmol, aromaticity=self._aromaticity)
        self.rkmol = Chem.AddHs(rkmol, explicitOnly=False, addCoords=True)
        return

    def _finish(self, **kwargs):
        # finish construction clean-ups

        props = self.rkmol.GetPropsAsDict()
        for k in props.keys():
            self.rkmol.ClearProp(k)

        self.rkmol.UpdatePropertyCache(True)
        self.rkmol.RemoveAllConformers()

        # sanitize again
        self.rkmol = rkutil.sanitize_rkmol(self.rkmol, aromaticity=self._aromaticity)
        Chem.Cleanup(self.rkmol)

        assert self._natoms == len(self._atomic_numbers) == len(self._atomic_masses) \
              == len(self._atomic_symbols) == len(self._formal_charges) \
                == self.rkmol.GetNumAtoms(onlyExplicit=False)

        if len(self._conformers) > 0:
            for conf in self.conformers:
                assert self._natoms == conf.coords.shape[0]

        if self.name is None or len(self.name) == 0:
            mapped_smiles = self.get_mapped_smiles(isomeric=False)
            # use md5 to ensure reproducibility
            m = hashlib.md5()
            m.update(mapped_smiles.encode('utf-8'))
            self.name = rkutil.get_mol_formula(self.rkmol) + '_' + m.hexdigest()
        return

    def get_bonds(self) -> List[Tuple]:
        '''get bond list (i,j) and i<j'''
        bonds = [tuple(sorted([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])) for bond in self.rkmol.GetBonds()]
        bonds.sort(key=itemgetter(0, 1))
        return bonds

    def get_mapped_smiles(self,
                          smarts_atoms: Union[Dict, Iterable[int]] = None,
                          *,
                          isomeric: bool = True,
                          kekulize: bool = False) -> str:
        ''' this always preserves chirality if existing in rkmol.
            If smarts_atoms is given, adding number to selected atoms,
            else, adding number by atom index + 1
        '''
        _rkmol = Chem.Mol(self.rkmol)  # make a copy
        rkutil.add_atom_map_num(_rkmol, smarts_atoms)
        # according to rdkit document
        # this includes H atoms since all H atoms are explicit in self.rkmol
        return Chem.MolToSmiles(_rkmol,
                                isomericSmiles=isomeric,
                                kekuleSmiles=kekulize,
                                canonical=True,
                                allBondsExplicit=False,
                                allHsExplicit=True,
                                doRandom=False)

    def get_smiles(self, *, isomeric: bool = True, kekulize: bool = False) -> str:
        '''returns canonical smiles without H atoms.'''
        return rkutil.get_smiles(self.rkmol, isomeric=isomeric, kekulize=kekulize)

    def merge(self, other: "Molecule", check_chiral: bool = True):
        '''merge other molecule into this molecule.'''

        if not self.get_mapped_smiles(isomeric=check_chiral) == other.get_mapped_smiles(isomeric=check_chiral):
            # mapped smiles must be identical, otherwise moledata and conformers mismatch
            raise ValueError('only identical molecules can be merged.')

        # self.rkmol unchanged
        # self.name unchanged
        # self.moledata updated, existing values unchanged
        for k, v in other.moledata.items():
            if k not in self.moledata:
                self.moledata[k] = v

        # conformers are combined
        self._conformers.extend(other._conformers)

        self._finish()

        return

    ################################################################################
    # BASIC IO routines
    # support only sdf, xyz and extxyz formats
    # we DO NOT use the metadata fields in sdf format for simplicity
    # all molecule data and atom data are written to extxyz formats for simple cases
    ################################################################################

    def _parse_conformers(self):
        '''parse conformers in self.rkmol, remove all conformers from self.rkmol after parsing'''
        assert not self._conformers
        rkconfs = self.rkmol.GetConformers()
        if len(rkconfs) == 0:
            return

        # at this point, self.rkmol should have all atoms EXPLICIT
        natoms = self.rkmol.GetNumAtoms(onlyExplicit=False)
        atomic_numbers = [at.GetAtomicNum() for at in self.rkmol.GetAtoms()]
        for conf in rkconfs:
            coords = np.vstack([conf.GetAtomPosition(idx) for idx in range(natoms)])
            conformer = Conformer(coords, atomic_numbers)
            confprops = conf.GetPropsAsDict()
            for k, v in confprops.items():
                prop = prefix_prop_key(k)
                conformer.confdata[prop] = v
            self.append_conformers(conformer)
        return

    def _from_rkmol(self, rkmol: Chem.Mol, keep_conformers: bool = True, keep_mol_prop: bool = False, **kwargs):
        self.rkmol = copy.deepcopy(rkmol)
        rkutil.clear_atom_map_num(self.rkmol)

        # sanitize first
        self._sanitize()
        self._normalize()

        # after sanitize, all atoms (hydrogens) are explicit and have valid coordinates if nconfs > 0
        # setup intrinsic moledata
        self._atomic_numbers = [atom.GetAtomicNum() for atom in self.rkmol.GetAtoms()]
        self._atomic_symbols = [atom.GetSymbol() for atom in self.rkmol.GetAtoms()]
        self._atomic_masses = [atom.GetMass() for atom in self.rkmol.GetAtoms()]
        self._formal_charges = [atom.GetFormalCharge() for atom in self.rkmol.GetAtoms()]
        self._nconfs = len(self._conformers)
        self._natoms = len(self._atomic_numbers)

        # conf generation relies on proper stereo chemistry, so must be placed here
        self.rkmol = rkutil.cleanup_rkmol_stereochemistry(self.rkmol)
        nconfs = kwargs.pop('nconfs', 0)
        if nconfs > 0:
            self.rkmol, _, _ = rkutil.generate_confs(self.rkmol, nconfs=nconfs, **kwargs)
            # the generated may be different from the requested, so we have to cleanup again
            self.rkmol = rkutil.cleanup_rkmol_stereochemistry(self.rkmol)
        molprops = rkmol.GetPropsAsDict() if keep_mol_prop else {}

        if keep_conformers:
            self._parse_conformers()
            for conf in self.conformers:
                for k, v in molprops.items():
                    prop = prefix_prop_key(k)
                    conf.confdata[prop] = v
        else:
            for k, v in molprops.items():
                prop = prefix_prop_key(k)
                self.moledata[prop] = v

        return

    def _from_smiles(self, smiles: str, nconfs: int = 0, **kwargs):
        '''construct a molecule without conformers'''
        rkmol = rkutil.get_mol_from_smiles(smiles)
        if nconfs > 0:
            kwargs['nconfs'] = nconfs
        self._from_rkmol(rkmol, keep_conformers=True, **kwargs)
        return

    def _from_mapped_smiles(self, smiles: str, nconfs: int = 0, **kwargs):
        '''construct a molecule without conformers from mapped smiles. H atoms must exist in mapped smiles.'''
        rkmol = rkutil.get_mol_from_smiles(smiles)
        # reassign atom idx according to atom map number
        rkmol = rkutil.renumber_atoms_with_atom_map_num(rkmol)
        if nconfs > 0:
            kwargs['nconfs'] = nconfs
        self._from_rkmol(rkmol, keep_conformers=True, **kwargs)
        return

    def _from_sdf(self, filename: str, keep_conformers: bool = True, **kwargs):
        '''read the first frame from the sdf file'''
        suppl = Chem.SDMolSupplier(filename, sanitize=False, removeHs=False)
        self._from_rkmol(suppl[0], keep_conformers=keep_conformers, **kwargs)
        return

    def _from_xyz(self, filename: str, mapped_smiles: str = None, **kwargs):
        # TODO: if the xyz file contains frames with mixed up atoms, this function will fail
        # atoms_list = aseio.read(filename, index=':', format="extxyz")
        atoms_list = aseio.read(filename, index=':', format="extxyz", properties_parser=xyz_properties_parser)

        if "name" in atoms_list[0].info and not self.name:
            self.name = atoms_list[0].info["name"]

        if "mapped_smiles" in atoms_list[0].info:
            mapped_smiles = atoms_list[0].info["mapped_smiles"]
        if mapped_smiles is None:
            raise MappedSmilesMissingException("'mapped_smiles' missed in xyz file.")

        # this conformer is only used to infer stereochemistry
        rkmol = rkutil.get_mol_from_smiles(mapped_smiles)
        rkmol = rkutil.renumber_atoms_with_atom_map_num(rkmol)
        rkmol = rkutil.append_conformers_to_mol(rkmol, [atoms_list[0].positions])
        keep_conformers = kwargs.pop("keep_conformers")
        self._from_rkmol(rkmol, keep_conformers=False, **kwargs)

        for atoms in atoms_list:
            atoms.info.pop("mapped_smiles", None)

        # detect common data in all confs and move to moledata
        common_info = set([(k, v) for k, v in atoms_list[0].info.items()])
        for i in range(1, len(atoms_list)):
            common_info = common_info & set([(k, v) for k, v in atoms_list[i].info.items()
                                            ])  # find identical (k,v) pairs

        for k, v in common_info:
            if is_charge_key(k) or is_energy_key(k) or is_force_key(k) or is_smiles_key(k):
                continue
            elif is_topo_params_key(k):
                self.moledata[k] = TopoParams.loads(v)
            else:
                if isinstance(v, str):
                    try:
                        self.moledata[k] = literal_eval(v)
                    except (SyntaxError, TypeError, ValueError):
                        self.moledata[k] = v
                elif isinstance(v, (int, float, np.integer)):
                    self.moledata[k] = v
                for atoms in atoms_list:
                    atoms.info.pop(k)

        if keep_conformers:
            conf_list = [Conformer.from_ase_atoms(atoms) for atoms in atoms_list]
            self.append_conformers(conf_list)
        return

    ########################################
    # constructors
    ########################################

    @classmethod
    def from_rdkit(cls, rkmol, **kwargs):
        '''H atoms are always explicitly added with 3d coords'''
        return cls(rkmol, **kwargs)

    @classmethod
    def from_smiles(cls, smiles: str, nconfs: int = 0, **kwargs):
        '''H atoms are always explicitly'''
        kwargs.pop('fmt', None)
        kwargs['nconfs'] = nconfs
        return cls(smiles, fmt='smiles', **kwargs)

    @classmethod
    def from_mapped_smiles(cls, smiles: str, nconfs: int = 0, **kwargs):
        '''H atoms are always explicitly'''
        kwargs.pop('fmt', None)
        kwargs['nconfs'] = nconfs
        return cls(smiles, fmt='mapped_smiles', **kwargs)

    @classmethod
    def from_sdf(cls, filename: str, **kwargs):
        '''H atoms are always explicitly added with 3d coords'''
        kwargs.pop('fmt', None)
        return cls(filename, fmt='sdf', **kwargs)

    @classmethod
    def from_xyz(cls, filename: str, mapped_smiles: str = None, **kwargs):
        '''H atoms are always explicitly added with 3d coords'''
        kwargs.pop("fmt", None)
        return cls(filename, fmt="xyz", mapped_smiles=mapped_smiles, **kwargs)

    from_extxyz = from_xyz

    ########################################
    # writers
    ########################################
    def to_rkmol(self, conf_id: int = None) -> Chem.Mol:
        rkmol = Chem.Mol(self.rkmol)  # deep copy
        rkmol.RemoveAllConformers()

        if conf_id is not None:
            assert -self.nconfs <= conf_id < self.nconfs
            conf_list = [self.conformers[conf_id]]
        else:
            conf_list = self.conformers
        rkmol = rkutil.append_conformers_to_mol(rkmol, [conf.coords for conf in conf_list])

        for k, v in self.moledata.items():
            if k.startswith('prop_'):
                rkmol.SetProp(k, str(v))
        return rkmol

    def to_sdf(self, filename: str, conf_id: int = None, append=False) -> int:
        rkmol = self.to_rkmol(conf_id=conf_id)
        if conf_id is not None:
            assert -self.nconfs <= conf_id < self.nconfs
            conf_list = [self.conformers[conf_id]]
        else:
            conf_list = self.conformers
        if not filename.endswith('sdf'):
            filename = filename + '.sdf'

        mode = 'a' if append else 'w'
        with open(filename, mode) as fout:
            with Chem.SDWriter(fout) as w:
                for i, conf in enumerate(conf_list):
                    confdata = conf.confdata
                    # add conf prop to rkmol for writing
                    for k, v in confdata.items():
                        if k.startswith('prop_'):
                            rkmol.SetProp(k, str(v))
                    w.write(rkmol, confId=i)
                    # remove conf prop for next
                    for k, v in confdata.items():
                        if k.startswith('prop_'):
                            rkmol.ClearProp(k)

        return len(conf_list)

    def to_xyz(self, filename: str, conf_id=None, append=False, confkeys=None):
        if conf_id is not None:
            assert -self.nconfs <= conf_id < self.nconfs
            conf_list = [self.conformers[conf_id]]
        else:
            conf_list = self.conformers
        if confkeys is not None and "mapped_smiles" not in confkeys:
            confkeys = list(confkeys) + ["mapped_smiles"]

        moledata = self.moledata.copy()
        moledata['name'] = self.name
        moledata['mapped_smiles'] = self.get_mapped_smiles(isomeric=True)

        hessian = []
        for conf in conf_list:
            for key in moledata:
                assert key not in conf.confdata, f"{key}"
            conf.confdata.update(moledata)
            hessian.append(conf.confdata.pop("hessian", None))
        write_conformers_to_extxyz(conf_list, filename, append=append, confkeys=confkeys)
        for i, conf in enumerate(conf_list):
            for key in moledata:
                conf.confdata.pop(key)
            if hessian[i] is not None:
                conf.confdata["hessian"] = hessian[i]
        return len(conf_list)

    to_extxyz = to_xyz

    def to_pdb(self, filename: str, conf_id: int = 0, multiple_connect: bool = True, no_connect: bool = False):
        ''' save Molecule to pdb file
            filename:
            conf_id: conformer index for saving pdb
            multiple_connect: use multiple CONECTs to encode bond order
            no_connect: Don't write any CONECT records
        '''
        rkmol = self.to_rkmol(conf_id=conf_id)
        flavor = 0
        if no_connect:
            flavor = 2
        elif not multiple_connect:
            flavor = 8
        Chem.MolToPDBFile(rkmol, filename, confId=0, flavor=flavor)

    def to_image(self,
                 filename: str = None,
                 size=(400, 300),
                 highlight: Iterable = None,
                 remove_h: bool = False,
                 plot_kekulize: bool = False,
                 idx_base_1: bool = False) -> Image:

        img = rkutil.show_mol(self.rkmol,
                              size=size,
                              highlight=highlight,
                              remove_h=remove_h,
                              plot_kekulize=plot_kekulize,
                              idx_base_1=idx_base_1)
        if filename is not None:
            img.save(filename)
        return img

    def copy(self, keep_conformers=True):
        rkmol = Chem.Mol(self.rkmol)
        if keep_conformers and self.nconfs > 0:
            rkmol = rkutil.append_conformers_to_mol(rkmol, [self.conformers[0].coords])
        copy_mol = Molecule(rkmol, name=self.name, keep_conformers=False)
        copy_mol.moledata = copy.deepcopy(self.moledata)
        if keep_conformers:
            copy_mol._conformers = [conf.copy() for conf in self.conformers]
        return copy_mol

    ################################################################################
    # modify conformations
    ################################################################################

    def append_conformers(self, conformers: Union[List[Conformer], Conformer]):
        if isinstance(conformers, Conformer):
            conformers = [conformers]
        first_conf = self.get_conformer() if self.nconfs else None
        for conf in conformers:
            assert tuple(conf.symbols) == tuple(self.atomic_symbols)
            if first_conf is None:
                first_conf = conf
            else:
                assert set(first_conf.confdata.keys()) == set(conf.confdata.keys())
        self._conformers.extend(conformers)
        return

    def remove_conformer(self, conf_id: int):
        # delete a conformer from rkmol and its cooresponding molecule data and atom data
        assert 0 <= conf_id < self.nconfs
        self._conformers.pop(conf_id)
        return

    def update_confdata(self, confdata: dict):
        for key, value in confdata.items():
            assert isinstance(value, Iterable)
            assert len(value) == self.nconfs
            for i, conf in enumerate(self._conformers):
                conf.confdata[key] = value[i]
        return

    ################################################################################
    # read information
    ################################################################################

    ################################################################################
    # molecule-level information
    ################################################################################
    def get_molecule_mass(self) -> float:
        '''return in a.u.'''
        return sum(self.atomic_masses)

    def get_molecule_formula(self) -> str:
        '''return the chemical formula'''
        return rkutil.get_mol_formula(self.rkmol)

    def get_moledata(self, prop: str) -> List:
        '''get {prop} from self.moledata'''
        if not prop in self.moledata:
            raise KeyError(f"molecule property {prop} is not in moledata")
        return self.moledata[prop]

    def get_rkmol(self) -> Chem.Mol:
        '''get a copy of self.rkmol, this does not include conformers'''
        return copy.deepcopy(self.rkmol)

    ################################################################################
    # atom-level information
    ################################################################################
    @property
    def aromaticity(self) -> str:
        return self._aromaticity

    @property
    def natoms(self) -> int:
        return self._natoms

    @property
    def atomic_masses(self) -> List[float]:
        return self._atomic_masses

    @property
    def atomic_numbers(self) -> List[int]:
        return self._atomic_numbers

    @property
    def atomic_symbols(self) -> List[str]:
        return self._atomic_symbols

    @property
    def formal_charges(self) -> List[int]:
        return self._formal_charges

    ################################################################################
    # conformer information
    ################################################################################
    @property
    def nconfs(self) -> int:
        return len(self._conformers)

    @property
    def conformers(self) -> List[Conformer]:
        return self._conformers.copy()

    def get_conformer(self, index: int = 0) -> Conformer:
        return self._conformers[index]

    def _get_all_confdata(self) -> dict:
        if self.nconfs == 0:
            return dict()
        conf = self.get_conformer()
        return {k: [conf.confdata[k] for conf in self.conformers] for k in conf.confdata}

    def get_confdata(self, prop: str):
        confdata = self._get_all_confdata()
        if not prop in confdata:
            raise KeyError("conformer property {} is not in confdata".format(prop))
        return confdata[prop]

    ################################################################################
    # modify mol/conf props
    ################################################################################

    def set_mol_prop(self, prop: str, value):
        # raise DeprecationWarning('avoid using props in sdf.')
        prop = prefix_prop_key(prop)
        self.moledata[prop] = value
        return

    def set_conf_prop(self, prop: str, value, conf_id: int = 0):
        # raise DeprecationWarning('avoid using props in sdf.')
        assert -self.nconfs < conf_id < self.nconfs
        prop = prefix_prop_key(prop)
        self.conformers[conf_id].confdata[prop] = value
        return

    def get_mol_prop(self, prop: str):
        # raise DeprecationWarning('avoid using props in sdf.')
        prop = prefix_prop_key(prop)
        return self.moledata[prop]

    def get_conf_prop(self, prop: str, conf_id: int = 0):
        # raise DeprecationWarning('avoid using props in sdf.')
        assert -self.nconfs < conf_id < self.nconfs
        prop = prefix_prop_key(prop)
        return self.conformers[conf_id].confdata[prop]

    def set_partial_charges(self, partial_charges: Iterable, partial_charges_key: str = 'partial_charges'):
        assert is_charge_key(partial_charges_key), 'partial charges key name must end with _charge or _charges'
        assert self.nconfs > 0

        partial_charges = np.asarray(partial_charges)

        if partial_charges.ndim == 1:
            assert partial_charges.shape[0] == self.natoms, 'partial_charges does not match natoms'
            for conf in self.conformers:
                conf.confdata[partial_charges_key] = np.asarray(partial_charges)
        elif partial_charges.ndim == 2:
            assert partial_charges.shape[0] == self.nconfs, 'partial_charges does not match nconfs'
            assert partial_charges.shape[1] == self.natoms, 'partial_charges does not match natoms'
            for i, conf in enumerate(self.conformers):
                conf.confdata[partial_charges_key] = np.asarray(partial_charges[i])
        else:
            raise ValueError(
                f'partial charges shape mismatch. expect ({self.nconfs}, {self.natoms}), got ({partial_charges.shape})')
        return

    def get_partial_charges(self, conf_id: int = 0, partial_charges_key: str = 'partial_charges'):
        return self.conformers[conf_id].confdata[partial_charges_key]


def read_molecules_from_sdf(sdf_file: str,
                            skip_error: bool = True,
                            check_chiral: bool = False,
                            keep_mol_prop: bool = False) -> Dict[str, Molecule]:
    '''read a multi-frame sdf file, isomeric molecules with identical atom indices are merged to multi conformers,
       sdf props are interpreted as conformer data
    '''
    suppl = Chem.SDMolSupplier(sdf_file, sanitize=False, removeHs=False)
    result = OrderedDict()
    for rkmol in suppl:
        try:
            cur_mol = Molecule.from_rdkit(rkmol, keep_mol_prop=keep_mol_prop)
            cur_smi = cur_mol.get_mapped_smiles(isomeric=check_chiral)
            if cur_smi not in result:
                result[cur_smi] = cur_mol
            else:
                result[cur_smi].merge(cur_mol, check_chiral=check_chiral)
        except (ValueError, AssertionError) as e:
            traceback.print_exc()
            if skip_error:
                continue
            else:
                raise AssertionError('parsing molecule from sdf file failed') from e

    return result


def read_molecules_from_xyz(xyz_file: str, check_chiral: bool = True) -> Dict[str, Molecule]:
    atoms_list = aseio.read(xyz_file, index=':', format="extxyz", properties_parser=xyz_properties_parser)
    smi_to_atoms_list = defaultdict(list)
    for atoms in atoms_list:
        mapped_smiles = atoms.info["mapped_smiles"]
        rkmol = rkutil.get_mol_from_smiles(mapped_smiles)
        rkmol = rkutil.renumber_atoms_with_atom_map_num(rkmol)
        rkmol = rkutil.append_conformers_to_mol(rkmol, [atoms.positions])
        mol = Molecule(rkmol, keep_conformers=False)
        smi = mol.get_mapped_smiles(isomeric=check_chiral)
        smi_to_atoms_list[smi].append(atoms)

    result = OrderedDict()
    for smi, atoms_list in smi_to_atoms_list.items():
        mol = Molecule.from_mapped_smiles(smi)
        conf_list = [Conformer.from_ase_atoms(atoms) for atoms in atoms_list]
        mol.append_conformers(conf_list)
        result[smi] = mol
    return result


def assert_good_molecule(mol: Molecule) -> None:
    '''
    a 'good' molecule satisfy:
    1. is an individual molecule, '.' does not appear in smiles
    2. elements covered by [H, C, N, O, F, P, S, Cl, Br, I]
    3. formal charge for [F, Cl, Br, I] must be negative
    4. halogen connectivity must be 1
    5. hybridization in S, SP, SP2, SP3

    '''
    smi = mol.get_smiles(isomeric=True)
    assert '.' not in smi, f'not an individual molecule: {smi}'

    good_elements = set([1, 6, 7, 8, 9, 15, 16, 17, 35, 53])
    assert set(mol.atomic_numbers).issubset(good_elements), f'bad elements found: {mol.atomic_numbers}'

    for num, fc in zip(mol.atomic_numbers, mol.formal_charges):
        if num in [9, 17, 35, 53]:
            assert fc <= 0, f'bad formal charge found: element {num} formal charge {fc}'

    connectivity = Counter()
    for ai, aj in mol.get_bonds():
        connectivity[ai] += 1
        connectivity[aj] += 1

    for idx, num in enumerate(mol.atomic_numbers):
        if num in [9, 17, 35, 53]:
            assert connectivity[idx] == 1, f'bad connectivity: idx {idx} atomic number {num} is X{connectivity[idx]}'

    allow_hb = {
        Chem.HybridizationType.S,
        Chem.HybridizationType.SP,
        Chem.HybridizationType.SP2,
        Chem.HybridizationType.SP3,
    }
    rkmol = mol.get_rkmol()
    assert NumRadicalElectrons(rkmol) == 0, 'found a radical'
    for at in rkmol.GetAtoms():
        hb = at.GetHybridization()
        assert hb in allow_hb, f'bad hybridization {hb} of atom {at}'

    return
