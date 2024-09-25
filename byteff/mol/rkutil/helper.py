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
from typing import Iterable, List, Tuple

from rdkit import Chem

logger = logging.getLogger(__name__)


def sorted_tuple(atom_tuple: Tuple[int]) -> Tuple[int]:
    ''' put smaller numbers first, to ensure unique representation for equivalent
        bonds, angles and dihedrals.
        e.g., (3,1,2)->(2,1,3), (4,3,2,1)->(1,2,3,4), (1,3,2,1)->(1,2,3,1)
        the last case may not occur in our program.
    '''
    for i in range(len(atom_tuple) // 2):
        if atom_tuple[i] > atom_tuple[-i - 1]:
            return tuple(atom_tuple[::-1])
        elif atom_tuple[i] < atom_tuple[-i - 1]:
            return tuple(atom_tuple)
    return tuple(atom_tuple)


def sorted_atomids(atomids: Iterable[int], is_improper: bool = False) -> Tuple:
    atomids = list(atomids)
    if is_improper:
        assert len(atomids) == 4
        return tuple([atomids[0]] + sorted(atomids[1:]))
    else:
        assert 0 < len(atomids) < 5
        if len(atomids) == 1:
            return tuple(atomids)
        return sorted_tuple(atomids)
