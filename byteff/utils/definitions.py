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

import json
from enum import Enum
from typing import Union

import torch

from .nested_data import NestedData

fudgeLJ = 0.5
fudgeQQ = 0.8333333333

# from gromacs document
# The electric conversion factor f=1/(4 pi eps0)=
# 138.935458 kJ mol−1 nm e−2. = 332.0637141491396 kcal mol−1 A e−2.
# chg for 'charge'
CHG_FACTOR = 332.0637141491396

MAX_TOTAL_CHARGE = 5
MAX_FORMAL_CHARGE = 2
MAX_RING_SIZE = 8
PROPERTORSION_TERMS = 4

supported_atomic_number = [1, 6, 7, 8, 9, 15, 16, 17, 35, 53]  # H, C, N, O, F, P, S, Cl, Br, I
atomic_number_map = {at: i for i, at in enumerate(supported_atomic_number)}


class BondOrder(Enum):
    single = 1.
    double = 2.
    triple = 3.
    aromatic = 1.5


class TopoTerm(Enum):
    """Possible topology terms for classical force field"""
    Atom = 1
    Bond = 2
    Angle = 3
    ProperTorsion = 4
    ImproperTorsion = 5
    Nonbonded14 = 6
    NonbondedAll = 7


class ParamTerm(Enum):
    """Possible parameter terms for classical force field"""
    Charge = 1
    Sigma = 2
    Epsilon = 3
    Bond_k = 4
    Bond_length = 5
    Angle_k = 6
    Angle_theta = 7
    ProperTorsion_k = 8
    ImproperTorsion_k = 9
    # ImproperPhase = 10


class TopoData(NestedData):
    DataTypeEnum = TopoTerm
    # Number of atoms for each TopoTerm and number of values for each parameter
    width: dict[TopoTerm, int] = {
        TopoTerm.Atom: 1,
        TopoTerm.Bond: 2,
        TopoTerm.Angle: 3,
        TopoTerm.ProperTorsion: 4,
        TopoTerm.ImproperTorsion: 4,
        TopoTerm.Nonbonded14: 2,
        TopoTerm.NonbondedAll: 2,
    }
    default_dtype: torch.dtype = torch.int32


class ParamData(NestedData):
    DataTypeEnum = ParamTerm
    # number of values for each parameter
    width: dict[ParamTerm, int] = {
        ParamTerm.Charge: 1,
        ParamTerm.Sigma: 1,
        ParamTerm.Epsilon: 1,
        ParamTerm.Bond_k: 1,
        ParamTerm.Bond_length: 1,
        ParamTerm.Angle_k: 1,
        ParamTerm.Angle_theta: 1,
        ParamTerm.ProperTorsion_k: PROPERTORSION_TERMS,
        ParamTerm.ImproperTorsion_k: 1,
        # ParamTerm.ImproperPhase: 1,
    }
    default_dtype: torch.dtype = torch.float32

    # gaff2, 100k data
    std_mean = {
        ParamTerm.Bond_k: (2.167e+02, 6.992e+02),
        ParamTerm.Bond_length: (1.905e-01, 1.282e+00),
        ParamTerm.Angle_k: (5.969e+01, 1.297e+02),
        ParamTerm.Angle_theta: (9.922e+00, 1.139e+02),
        ParamTerm.ProperTorsion_k: (1.188e+00, 4.226e-01),
        ParamTerm.ImproperTorsion_k: (3.711e+00, 4.291e+00),
        ParamTerm.Charge: (1.0, 2.052e-03),
        ParamTerm.Sigma: (1.0, 2.920e+00),
        ParamTerm.Epsilon: (1.0, 7.258e-02),
    }


class TopoParams:
    """ Topology and parameters of classical forcefield for one molecule. """

    topo_param_map = {
        TopoTerm.Atom: [ParamTerm.Charge, ParamTerm.Sigma, ParamTerm.Epsilon],
        TopoTerm.Bond: [ParamTerm.Bond_k, ParamTerm.Bond_length],
        TopoTerm.Angle: [ParamTerm.Angle_k, ParamTerm.Angle_theta],
        TopoTerm.ProperTorsion: [ParamTerm.ProperTorsion_k],
        TopoTerm.ImproperTorsion: [ParamTerm.ImproperTorsion_k]
    }

    topo_param_map_rev: dict[ParamTerm, TopoTerm] = {}
    for k, v in topo_param_map.items():
        for term in v:
            topo_param_map_rev[term] = k

    def __init__(self,
                 topo: Union[TopoData, dict] = None,
                 param: Union[ParamData, dict] = None,
                 topo_dtype=torch.int32,
                 param_dtype=torch.float32):
        if isinstance(topo, TopoData):
            self.topo = topo
        else:
            self.topo = TopoData(topo, dtype=topo_dtype)
        if isinstance(param, ParamData):
            self.param = param
        else:
            self.param = ParamData(param, dtype=param_dtype)
        self.counts: dict[TopoTerm, torch.Tensor] = {term: None for term in TopoTerm}
        self.uncertainty: dict[ParamTerm, torch.Tensor] = {term: None for term in ParamTerm}

    def __getitem__(self, __key: Union[TopoTerm, ParamTerm]) -> torch.Tensor:
        if __key in self.topo.allowed_keys:
            return self.topo[__key]
        elif __key in self.param.allowed_keys:
            return self.param[__key]
        else:
            raise TypeError(f'{__key} is not supported by {type(self)}!')

    def __setitem__(self, __key: Union[TopoTerm, ParamTerm], __value: list):
        if __key in self.topo.allowed_keys:
            self.topo[__key] = __value
        elif __key in self.param.allowed_keys:
            self.param[__key] = __value
        else:
            raise TypeError(f'{__key} is not supported by {type(self)}!')

    def __str__(self) -> str:
        return f"Topology: {self.topo}\nParameters: {self.param}"

    def __repr__(self) -> str:
        return self.__str__()

    def get_count(self, term: TopoTerm):
        if self.counts[term] is None:
            topo = self[term]
            count = topo.shape[0]

            # check consistency
            if term in self.topo_param_map:
                for param_term in self.topo_param_map[term]:
                    assert self[param_term].shape[0] == count
            return torch.tensor([count], dtype=topo.dtype, device=topo.device)
        else:
            return self.counts[term]

    def dumps(self) -> str:
        data = {'topo': self.topo.to_dict(), 'param': self.param.to_dict()}
        return json.dumps(data)

    @classmethod
    def loads(cls, string: str):
        data = json.loads(string)
        topo = TopoData(data['topo'])
        param = ParamData(data['param'])
        return cls(topo=topo, param=param)

    def copy(self) -> "TopoParams":
        """ shallow copy """
        new_tp = TopoParams()
        for k, v in self.topo.items():
            new_tp.topo._data[k] = v
        for k, v in self.param.items():
            new_tp.param._data[k] = v
        for k, v in self.counts.items():
            new_tp.counts[k] = v
        return new_tp
