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

from enum import Enum
from typing import Union

import torch


class NestedData:
    """ A dict containing torch.Tensor with default shape """
    DataTypeEnum = Enum
    width: dict[Enum, int] = {}
    default_dtype: torch.dtype = torch.float32

    def _check_key(self, __key: DataTypeEnum):
        if isinstance(__key, self.DataTypeEnum):
            assert __key in self.allowed_keys, f'{self.allowed_keys}'
        else:
            raise TypeError(f'Type {type(__key)} is not supported by {type(self)}!')

    def _check_value(self, __key: DataTypeEnum, __value: Union[list, torch.Tensor]):
        if isinstance(__value, list):
            __value = torch.tensor(__value, dtype=self.dtype) if __value else self._default_value(__key)
        else:
            assert isinstance(__value, torch.Tensor)
            if __value.dtype != self.dtype:
                __value = __value.to(self.dtype)
        assert __value.shape[1] == self.width[__key], f'{__key}, {__value.shape}'
        return __value

    def _default_value(self, __key: DataTypeEnum):
        return torch.zeros((0, self.width[__key]), dtype=self.dtype)

    def __init__(self, data: dict = None, dtype: torch.dtype = None):
        self.allowed_keys = set(self.DataTypeEnum)
        self.dtype = self.default_dtype if dtype is None else dtype
        self.type = float if self.dtype.is_floating_point else int
        self._data = dict()
        data = {} if data is None else data
        for term in self.DataTypeEnum:
            v = data.get(term, None) or data.get(term.name, None)
            v = self._check_value(term, v) if v else self._default_value(term)
            self._data[term] = v

    def __getitem__(self, __key: DataTypeEnum) -> torch.Tensor:
        self._check_key(__key)
        return self._data[__key]

    def __setitem__(self, __key: DataTypeEnum, __value: Union[list, torch.Tensor]):
        self._check_key(__key)
        __value = self._check_value(__key, __value)
        self._data[__key] = __value

    def __str__(self) -> str:
        return str(self._data)

    def to_dict(self) -> dict[str, list]:
        data = {}
        for k, v in self._data.items():
            data[k.name] = v.detach().tolist()
        return data

    def items(self):
        return self._data.items()

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()
