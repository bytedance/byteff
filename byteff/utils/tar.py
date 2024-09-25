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

import io
import os
import tarfile
import typing as T
from pathlib import Path

import numpy as np


class TarFile(T.Sequence[bytes]):

    def __init__(self,
                 file_or_fp: T.Union[str, T.BinaryIO],
                 mode='r',
                 file_filter: T.Optional[T.AbstractSet[str]] = None,
                 verbose: bool = False):
        self._verbose = verbose
        if isinstance(file_or_fp, (str, Path)):
            self._file = tarfile.open(file_or_fp, mode)
        else:
            self._file = tarfile.open(fileobj=file_or_fp)
        self._fn = file_or_fp
        self._cache = None

        if file_filter is not None:
            self._members = {m.name for m in self.members if m.name in file_filter}
        else:
            self._members = {m.name for m in self.members}

    @property
    def members(self) -> T.List[tarfile.TarInfo]:
        if not hasattr(self, '_members'):
            return [m for m in self._file.getmembers() if m.isfile()]

        if self._cache is None:
            self._cache = [m for m in self._file.getmembers() if m.isfile() and m.name in self._members]

        return self._cache

    def __getitem__(self, index: T.Union[int, str]) -> bytes:
        if isinstance(index, int):
            fn = self.members[index]
        elif isinstance(index, str):
            if index not in self._members:
                raise KeyError(f'invalid key {index}')
            fn = index
        else:
            raise KeyError(f'invalid key type {type(index)}')

        return self._file.extractfile(fn).read()

    def __contains__(self, item: T.Union[int, str]):
        return item in self._members if isinstance(item, str) else 0 <= item < len(self)

    def __len__(self) -> int:
        # can have duplicate keys inside members
        # len(self._members) <= len(self.members)
        return len(self.members)

    def write(self, data: bytes, key: str = ''):
        if self._file.mode[0] == 'r':
            raise IOError("cannot write tar file in readonly mode")

        if len(key) > 0:
            info = self._file.tarinfo(key)
        else:
            info = self._file.tarinfo(str(len(self._file.getmembers())))

        info.size = len(data)
        self._file.addfile(info, fileobj=io.BytesIO(data))
        # no need to take filter into consideration during writing stage
        self._members.add(key)
        # flush cache
        self._cache = None

    def close(self):
        return self._file.close()

    def __enter__(self):
        if self._file.fileobject.closed:
            if isinstance(self._fn, (str, Path)):
                self._file = tarfile.open(self._fn, self._file.mode)
            else:
                raise IOError("file has been closed")

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class MultiTarFileReader(T.Sequence[bytes]):

    def __init__(self,
                 files_or_fps: T.Union[T.List[str], T.BinaryIO],
                 mode='r',
                 filters: T.Optional[T.Dict[str, T.AbstractSet[str]]] = None,
                 verbose: bool = False):
        """
        filters should be in a mapping form which maps a tar file name to corresponding members
        """
        if mode[0] == 'w' or mode[0] == 'a':
            raise ValueError("MultiTarFileReader is a readonly interface")
        if filters is None:
            filters = {}
        self.files = [
            TarFile(fn, mode, file_filter=filters.get(fn if isinstance(fn, str) else idx, None), verbose=verbose)
            for idx, fn in enumerate(files_or_fps)
        ]
        self.filemap = {(fn if isinstance(fn, str) else idx): self.files[idx] for idx, fn in enumerate(files_or_fps)}
        self._verbose = verbose
        self._file_offsets = np.cumsum([len(f) for f in self.files])

    def __len__(self):
        return sum(len(f) for f in self.files)

    def __getitem__(self, item):
        if isinstance(item, str):
            for f in self.files:
                if item in f:
                    return f[item]
        elif isinstance(item, int):
            cur = item
            idx = self._file_offsets.searchsorted(cur, 'right')

            if idx >= len(self.files):
                raise KeyError(f"index {item} is out of bounds")
            return self.files[idx][cur - self._file_offsets[idx].item()]
        elif isinstance(item, tuple) or isinstance(item, list):
            tarfn, fn = item
            if tarfn in self.filemap:
                return self.filemap[tarfn][fn]
            else:
                raise KeyError(f'tarfile {tarfn} does not exist')
        else:
            raise TypeError("we only support (tarfile, file) or str/int indexing")

    def __iter__(self):
        for f in self.files:
            yield from iter(f)


class MultiTarFileWithMeta:

    def __init__(self,
                 data_dir: str,
                 mode="r",
                 meta_file="meta.txt",
                 tar_dir="tar_files",
                 write_size=10.,
                 name_filter: T.Sequence[str] = None) -> None:
        """
        A lazy reader for multiple tar files with one meta data.
        write_size in MB 
        """
        self.data_dir = data_dir
        self.meta_file = meta_file
        self.tar_dir = tar_dir
        self.mode = mode
        self.name_filter = set(name_filter) if name_filter is not None else None

        if mode == "r":
            with open(os.path.join(data_dir, meta_file)) as file:
                lines = file.readlines()
            self.meta_info = [s.rstrip().split(",") for s in lines]
            if self.name_filter is not None:
                self.meta_info = [info for info in self.meta_info if info[1].split(".")[0] in self.name_filter]
            self.name_idx = {info[1].split(".")[0]: idx for idx, info in enumerate(self.meta_info)}
        else:
            os.makedirs(os.path.join(data_dir, tar_dir), exist_ok=True)
            self.meta_info = []
            self.name_idx = {}

        self._tar_files: T.Dict[str, TarFile] = {}
        self._tar_sizes: T.Dict[str, int] = {}

        # size for each tar file when writing
        self.write_size = write_size

    def __len__(self):
        return len(self.meta_info)

    def get_index(self, _key: str) -> int:
        return self.name_idx.get(_key, None)

    def _post_process(self, data):
        return data

    def __getitem__(self, item: T.Union[str, int, slice]) -> T.Dict[str, bytes]:
        """ Return a dict of data given index or name.
            Return None if no such data.
        """
        if item is None:
            return None
        elif isinstance(item, str):
            index = self.get_index(item)
            return self.__getitem__(index)
        elif isinstance(item, int):
            if item >= len(self.meta_info):
                return None
            info = self.meta_info[item]
            if info[0] not in self._tar_files:
                self._tar_files[info[0]] = TarFile(os.path.join(self.data_dir, self.tar_dir, info[0]))
            ret = dict()
            for name in info[1:]:
                ret[name] = self._tar_files[info[0]][name]
            return self._post_process(ret)
        elif isinstance(item, slice):
            start, stop, step = item.indices(len(self))
            return (self[i] for i in range(start, stop, step))
        else:
            raise TypeError("Only str, int and slice are supported.")

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def write(self, data: T.Dict[str, bytes]):

        def new_tar(name):
            self._tar_files[name] = TarFile(os.path.join(self.data_dir, self.tar_dir, name), mode="w")
            self._tar_sizes[name] = 0

        def cur_name():
            return f"{len(self._tar_files) - 1}.tar"

        assert self.mode == "w"

        # create a new tarfile if overflow
        if not self._tar_files or (self._tar_sizes[cur_name()] > self.write_size):
            new_tar(f"{len(self._tar_files)}.tar")

        for k, v in data.items():
            self._tar_files[cur_name()].write(v, k)
            self._tar_sizes[cur_name()] += len(v) / 1e6
        self.meta_info.append([cur_name()] + list(data.keys()))
        name = list(data.keys())[0].split(".")[0]
        self.name_idx[name] = len(self.meta_info) - 1

    def write_meta(self, save_path=""):
        assert self.mode == "w"
        lines = [",".join(info) + "\n" for info in self.meta_info]
        if not save_path:
            save_path = os.path.join(self.data_dir, self.meta_file)
        with open(save_path, "w") as file:
            file.writelines(lines)

    def close(self):
        for v in self._tar_files.values():
            v.close()
        if self.mode == "w":
            self.write_meta()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
