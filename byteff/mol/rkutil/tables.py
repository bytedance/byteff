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

import typing as T

from rdkit import Chem

num_bondorder = {
    1: Chem.BondType.SINGLE,
    1.5: Chem.BondType.AROMATIC,
    "ar": Chem.BondType.AROMATIC,
    2: Chem.BondType.DOUBLE,
    3: Chem.BondType.TRIPLE,
}

periodic_table = Chem.GetPeriodicTable()

atomnum_elem: T.Dict[int, str] = {i: periodic_table.GetElementSymbol(i) for i in range(1, 119)}
atomnum_mass: T.Dict[int, float] = {i: periodic_table.GetAtomicWeight(i) for i in range(1, 119)}
elem_atomnum: T.Dict[str, int] = {elem: atomnum for atomnum, elem in atomnum_elem.items()}
elem_mass: T.Dict[str, float] = {elem: atomnum_mass[atomnum] for atomnum, elem in atomnum_elem.items()}


def get_atomnum_by_mass(mass: float) -> int:
    '''find the atomnum closest to mass'''
    retnum = 1
    dif = abs(mass - atomnum_mass[1])
    for anum, amass in atomnum_mass.items():
        adif = abs(mass - amass)
        if adif < dif:
            retnum = anum
            dif = adif
    return retnum
