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

from math import pi

from ase.units import Bohr, Hartree, kcal, kJ, mol, nm

# ========== Energy =============


def eV_to_kJ_mol(ev):
    '''
    energy conversion
    eV -> kJ/mol
    '''
    return ev / (kJ / mol)


def eV_to_kcal_mol(ev):
    '''
    energy conversion
    eV -> kcal/mol
    '''
    return ev / (kcal / mol)


def kj_to_kcal(value):
    '''
    energy conversion
    kJ/mol -> kcal/mol
    '''
    return value * kJ / kcal


def kcal_to_kj(value):
    '''
    energy conversion
    kcal/mol -> kJ/mol
    '''
    return value / kJ * kcal


def kcal_mol_to_eV(value):
    '''
    energy conversion
    kcal/mol -> eV
    '''
    return value * (kcal / mol)


def Hartree_to_eV(value):
    '''
    energy conversion
    Hartree -> eV
    '''
    return value * Hartree


def eV_to_Hartree(value):
    '''
    energy conversion
    eV -> Hartree
    '''
    return value / Hartree


def Hartree_to_kcal_mol(value):
    '''
    energy conversion
    Hartree -> kcal/mol
    '''
    return value * Hartree / (kcal / mol)


def kcal_mol_to_Hartree(value):
    '''
    energy conversion
    kcal/mol -> Hartree
    '''
    return value * (kcal / mol) / Hartree


kcal_to_kJ = kcal_to_kj
kJ_to_kcal = kj_to_kcal
kcal_mol_to_kJ_mol = kcal_to_kJ
kJ_mol_to_kcal_mol = kJ_to_kcal

# ========== Force ==============


def eV_A_to_kJ_mol_nm(eva):
    '''
    force conversion
    eV/A -> kJ/mol/nm
    '''
    return eva / (kJ / mol) * nm


def eV_A_to_kcal_mol_A(eva):
    '''
    force conversion
    eV/A -> kcal/mol/A
    '''
    return eva / (kcal / mol)


def kJ_mol_nm_to_kcal_mol_A(kj_mol_nm):
    '''
    force conversion
    kJ/mol/nm -> kcal/mol/A
    '''
    return kj_mol_nm / nm * kJ / kcal


def kcal_mol_A_to_eV_A(kcal_mol_a):
    '''
    force conversion
    kcal/mol/A -> eV/A
    '''
    return kcal_mol_a * (kcal / mol)


def Hartree_Bohr_to_eV_A(value):
    '''
    force conversion
    Hartree/Bohr -> eV/A
    '''
    return value * Hartree / Bohr


def eV_A_to_Hartree_Bohr(value):
    '''
    force conversion
    eV/A -> Hartree/Bohr
    '''
    return value / (Hartree / Bohr)


def Hartree_Bohr_to_kcal_mol_A(value):
    '''
    force conversion
    Hartree/Bohr -> kcal/mol/A
    '''
    return value * Hartree / Bohr / (kcal / mol)


def kcal_mol_A_to_Hartree_Bohr(value):
    '''
    force conversion
    kcal/mol/A -> Hartree/Bohr
    '''
    return value * (kcal / mol) / (Hartree / Bohr)


# ========== Distance ===========


def nm_to_A(value):
    '''
    distance conversion
    nm -> Angstrom
    '''
    return value * 10


def A_to_nm(value):
    '''
    distance conversion
    Angstrom -> nm
    '''
    return value / 10


def Bohr_to_A(value):
    '''
    distance conversion
    Bohr -> Angstrom
    '''
    return value * Bohr


def A_to_Bohr(value):
    '''
    distance conversion
    Angstrom -> Bohr
    '''
    return value / Bohr


# ========== Bond K =============


def kj_mol_nm2_to_kcal_mol_A2(value):
    '''
    bond k unit conversion
    kJ/mol/nm^2 -> kcal/mol/A^2
    '''
    return kJ_mol_to_kcal_mol(value) / 10 / 10


def kcal_mol_A2_to_kj_mol_nm2(value):
    '''
    bond k unit conversion
    kcal/mol/A^2 -> kJ/mol/nm^2
    '''
    return kcal_mol_to_kJ_mol(value) * 10 * 10


def Hartree_Bohr2_to_kcal_mol_A2(value):
    '''
    bond k unit conversion
    Hartree/Bohr^2 -> kcal/mol/A^2
    '''
    value = value * Hartree / Bohr / Bohr  # to eV/A^2
    return value / (kcal / mol)  # to kcal/mol/A^2


def kcal_mol_A2_to_Hartree_Bohr2(value):
    '''
    bond k unit conversion
    kcal/mol/A^2 -> Hartree/Bohr^2
    '''
    value = value * (kcal / mol)  # to eV/A^2
    return value / (Hartree / Bohr / Bohr)  # to Hartree/Bohr^2


kJ_mol_nm2_to_kcal_mol_A2 = kj_mol_nm2_to_kcal_mol_A2
kcal_mol_A2_to_kJ_mol_nm2 = kcal_mol_A2_to_kj_mol_nm2

# ========== Hessian =============


def hessian_to_s2_e24(value):
    '''
    hessian to freqs, step 1
    kcal/mol/(A^2 * g/mol) to 10^24 s^-2
    '''
    return kcal_to_kj(value) * 1000 / 10


def Hz_e12_to_cm_1(value):
    '''
    hessian to freqs, step 2
    10^12 Hz to cm-1
    '''
    return value * 0.5 / pi * 33.3564095


# ========== Angle =============


def degree_to_rad(value):
    '''
    Angle unit conversion
    Degree -> Radian
    '''
    return value / 180 * pi


def rad_to_degree(value):
    '''
    Angle unit conversion
    Radian -> Degree
    '''
    return value * 180 / pi
