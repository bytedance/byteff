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
import io
import logging
import re
from ast import literal_eval
from typing import List, Union

import ase
import ase.io as aseio
import ase.io.gromacs
import numpy as np
from ase.io.extxyz import key_val_str_to_dict as ase_properties_parser
from ase.io.extxyz import write_extxyz
from ase.symbols import Symbols

from byteff.utils import simple_unit as unit
from byteff.utils.definitions import TopoParams

logger = logging.getLogger(__name__)


def is_energy_key(key):
    assert isinstance(key, str)
    return key.endswith("energy") or key.endswith("energies")


def is_force_key(key):
    assert isinstance(key, str)
    return key.endswith("force") or key.endswith("forces")


def is_charge_key(key):
    assert isinstance(key, str)
    return key.endswith("charge") or key.endswith("charges")


def is_ids_key(key):
    assert isinstance(key, str)
    return key.endswith("id") or key.endswith("ids")


def is_smiles_key(key):
    assert isinstance(key, str)
    return key.endswith("smiles")


def is_multipole_key(key):
    assert isinstance(key, str)
    multipole_keys = ["dipole", "quadrupole", "octopole", "hexadecapole"]
    return any(key.endswith(multipole_key) for multipole_key in multipole_keys)


def is_topo_params_key(key):
    assert isinstance(key, str)
    return key.startswith("topo_params")


def prefix_prop_key(prop: str):
    if not prop.startswith('prop_'):
        prop = 'prop_' + prop
    return prop


def xyz_properties_parser(string, sep=None):
    smi_key = "mapped_smiles="
    if smi_key in string:
        prefix, suffix = string.split(smi_key)
        suffix_split = suffix.split(sep, maxsplit=1)
        suffix_split[0] = suffix_split[0].replace('\\', '\\\\')
        pattern = r"(?<!\\)\\n"
        string = ''.join([prefix, smi_key, ' '.join(suffix_split)])
        string = re.sub(pattern, r'\\\n', string)
    for pole in [" dipole=", " quadrupole=", " octopole=", " hexadecapole="]:
        string = string.replace(pole, " dft_" + pole[1:])
    return ase_properties_parser(string, sep=sep)


# A SINGLE conformation for a molecule with only geometry and atom data


class Conformer:

    def __init__(self, coords: np.array, symbols: List[str], confdata: dict = None):
        if not isinstance(coords, np.ndarray):
            coords = np.array(coords)
        assert coords.shape[0] == len(symbols)
        assert coords.shape[1] == 3
        self.natoms = coords.shape[0]
        if isinstance(symbols[0], int):
            symbols = list(Symbols(symbols))
        elif isinstance(symbols, Symbols):
            symbols = list(symbols)
        self.symbols = symbols

        confdata = confdata if confdata else dict()
        for confkey in confdata:
            if is_force_key(confkey):
                assert np.array(confdata[confkey]).shape == (self.natoms, 3)
            elif is_charge_key(confkey):
                assert np.array(confdata[confkey]).shape == (self.natoms,)
        self.confdata = confdata
        self.confdata["coords"] = coords

    @property
    def coords(self):
        return self.confdata["coords"]

    @coords.setter
    def coords(self, coords: np.array):
        assert isinstance(coords, np.ndarray) and coords.shape == (self.natoms, 3)
        self.confdata["coords"] = coords

    @classmethod
    def from_ase_atoms(cls, atoms: ase.Atoms, confdata: dict = None):
        coords = atoms.positions
        symbols = atoms.symbols
        confdata = dict() if confdata is None else confdata
        for key in atoms.arrays:
            if is_force_key(key):
                confdata[key] = unit.eV_A_to_kcal_mol_A(atoms.arrays[key])
            elif key not in ['numbers', 'positions']:
                confdata[key] = atoms.arrays[key]
        for key in atoms.info:
            if is_energy_key(key):
                confdata[key] = unit.eV_to_kcal_mol(atoms.info[key])
            elif isinstance(atoms.info[key], str):
                try:
                    confdata[key] = literal_eval(atoms.info[key])
                except (SyntaxError, TypeError, ValueError):
                    confdata[key] = atoms.info[key]
            else:
                confdata[key] = atoms.info[key]
        return cls(coords, symbols, confdata=confdata)

    @classmethod
    def from_xyz(cls, xyz_file: str):
        atoms_list = aseio.read(xyz_file, index=':', properties_parser=xyz_properties_parser)
        assert len(atoms_list) == 1, "Unable to load multi xyz confs to one Conformer object"
        return cls.from_ase_atoms(atoms_list[0])

    @classmethod
    def from_gromacs_gro(cls, gro_file: str):
        atoms = aseio.gromacs.read_gromacs(gro_file)
        return cls.from_ase_atoms(atoms)

    def to_ase_atoms(self, confkeys=None):
        atoms = ase.Atoms(self.symbols, positions=self.coords)
        if confkeys is None:
            confkeys = list(self.confdata.keys())
            confkeys.remove("coords")

        for key in confkeys:
            assert key in self.confdata, f"key {key} not in confdata keys {list(self.confdata.keys())}"

        # convert forces before other ase.Atoms.arrays keys
        for key in confkeys:
            if is_force_key(key):
                atoms.arrays[key] = unit.kcal_mol_A_to_eV_A(np.array(self.confdata[key]))

        for key in confkeys:
            data = self.confdata[key]
            if is_topo_params_key(key):
                assert isinstance(data, TopoParams)
                atoms.info[key] = data.dumps()
            elif is_force_key(key):
                continue
            elif is_energy_key(key):
                atoms.info[key] = unit.kcal_mol_to_eV(data)
            elif isinstance(data, (int, float)):
                atoms.info[key] = data
            elif isinstance(data, str):
                repr_data = repr(data)[1:-1]
                atoms.info[key] = repr_data
            elif isinstance(data, dict):
                atoms.info[key] = repr(data)
            elif np.iterable(data) and len(data) == self.natoms:
                atoms.arrays[key] = np.array(data)
            else:
                atoms.info[key] = repr(data)

        return atoms

    def to_xyz(self, filename: Union[str, io.StringIO], append=False, confkeys=None):
        # write to xyz format
        atoms = self.to_ase_atoms(confkeys=confkeys)
        io_flag = 'a' if append else 'w'
        if isinstance(filename, io.StringIO):
            write_extxyz(filename, atoms, append=append)
        else:
            with open(filename, io_flag) as f:
                write_extxyz(f, atoms, append=append)

    to_extxyz = to_xyz

    def copy(self):
        return Conformer(self.coords.copy(), copy.copy(self.symbols), confdata=copy.deepcopy(self.confdata))


def write_conformers_to_extxyz(conformers: List[Conformer],
                               filename: str,
                               append: bool = False,
                               confkeys: List[str] = None):
    with io.StringIO() as string_io:
        for conf in conformers:
            assert isinstance(conf, Conformer)
            conf.to_xyz(string_io, append=True, confkeys=confkeys)
        str_value = string_io.getvalue()

    with open(filename, 'a' if append else 'w') as f:
        f.write(str_value)
    return
