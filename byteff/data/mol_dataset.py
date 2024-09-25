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

import gc
import logging
import os
import random
from queue import Queue
from threading import Thread
from typing import Callable, Optional

from torch import LongTensor, Tensor
from torch_geometric.data import InMemoryDataset

from byteff.data.mol_data import MolData
from byteff.mol import Molecule, MolTarLoader
from byteff.mol.moltools import judge_mol_trainable
from byteff.utils.utilities import print_progress

logger = logging.getLogger(__name__)


class MolDataset(InMemoryDataset):

    def __init__(self,
                 root: str,
                 meta_file: str = None,
                 rank: int = 0,
                 world_size: int = 1,
                 shard_id: int = None,
                 shards: int = 1,
                 save_label: str = '',
                 process_shuffle: bool = False,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None,
                 process_only: bool = False,
                 **data_kwargs):

        self.data_kwargs = data_kwargs
        self.rank = rank
        self.world_size = world_size

        if shard_id is None:
            shard_ids = list(range(shards))
        elif isinstance(shard_id, int):
            assert 0 <= shard_id < shards
            shard_ids = [shard_id]
        else:
            assert all([0 <= s < shards and isinstance(s, int) for s in shard_id])
            shard_ids = list(shard_id)
        self.shard_ids = shard_ids[rank::world_size]
        self.shards = shards
        self.shard_data_num = []
        self._meta_file = meta_file
        self._save_label = save_label
        self._process_shuffle = process_shuffle
        super().__init__(root, transform=transform, pre_transform=pre_transform, pre_filter=pre_filter)

        self.finished = False

        if not process_only:
            self.load_shards()

    def len(self):
        if self.finished:
            return len(self._data_list)
        else:
            return super().len()

    @property
    def raw_file_names(self) -> str:
        return self._meta_file

    @property
    def raw_paths(self) -> str:
        # overwrite raw_paths
        return self._meta_file

    @property
    def processed_file_names(self) -> str:

        def get_name(shard_id):
            name = 'processed_data'
            name += f'_shard{shard_id}-{self.shards}'
            name += '' if not self._save_label else '_' + self._save_label
            return name + '.pt'

        names = [get_name(shard_id) for shard_id in self.shard_ids]
        return names

    def load_shards(self):
        self.finished = False
        self._data_list = []
        gc.collect()
        data_list = []
        for path in self.processed_paths:
            # self.load would set self._data_list to None automatically
            self.load(path)
            for data in self:
                data_list.append(data)
            self.shard_data_num.append(self.len())

        self._data_list = data_list
        self.finished = True
        self._data = None

    def save_shards(self):
        begin_idx = 0
        for path, num in zip(self.processed_paths, self.shard_data_num):
            self.save(self._data_list[begin_idx:begin_idx + num], path)
            begin_idx += num

    def process_single(self, mol: Molecule):
        return MolData.from_mol(mol, **self.data_kwargs)

    def process_batch(self, mol_loader: MolTarLoader, indices: list[int]) -> list[MolData]:

        count = 0
        data_list = []
        for i in indices:
            count += 1
            mol = mol_loader[i]
            if not judge_mol_trainable(mol):
                continue
            data = MolData.from_mol(mol, **self.data_kwargs)
            if data is not None:
                data_list.append(data)

            if self.shards == 1:
                print_progress(count, len(indices), "Converting")
            else:
                if count % 100 == 0:
                    logger.info(f"finished {count}")
        return data_list

    def process(self):
        # Read data into huge `Data` list.
        assert len(self.shard_ids) == 1, "shard_id is needed when process"
        mol_loader = MolTarLoader(os.path.dirname(self._meta_file), meta_file=os.path.basename(self._meta_file))
        indices = list(range(len(mol_loader)))[self.shard_ids[0]::self.shards]
        logger.info("Processing data. shard_id: %d, num mols: %d.", self.shard_ids[0], len(indices))
        data_list = self.process_batch(mol_loader, indices)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        if self._process_shuffle:
            random.shuffle(data_list)

        self.save(data_list, self.processed_paths[0])
        logger.info("Saved processed data to: %s", self.processed_paths[0])

    def update_single_data(self, new_data, index: int, attribute_name: str):
        setattr(self._data_list[index], attribute_name, new_data.clone())

    def update_data(self, batched_data: Tensor, counts: LongTensor, begin_index: int, attribute_name: str):
        assert self._indices is None
        assert counts.dim() == 1
        idx = 0
        batched_data = batched_data.to('cpu')
        for i, num in enumerate(counts):
            sliced_data = batched_data[idx:idx + num].clone()
            self.update_single_data(sliced_data, begin_index + i, attribute_name)
            idx += num

    def save_with_label(self, label: str):
        self._save_label = label
        self.save_shards()

    def load_with_label(self, label: str):
        if self._save_label == label:
            logger.info(f'skip loading label {label}')
            return
        current_label = self._save_label
        self._save_label = label
        if not all([os.path.exists(f) for f in self.processed_paths]):
            self._save_label = current_label
            raise FileNotFoundError(f"{self.processed_paths} does not exist!")
        else:
            self.load_shards()
