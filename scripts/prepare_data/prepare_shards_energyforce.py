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

import argparse
from concurrent.futures import ProcessPoolExecutor, wait

from byteff.data import LabelType, MolDataset
from byteff.utils import setup_default_logging

logger = setup_default_logging()

parser = argparse.ArgumentParser(description='prepare EnergyForce data to shards')
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--shards', type=int, default=4)
parser.add_argument('--meta_file', type=str)
parser.add_argument('--save_path', type=str)
args = parser.parse_args()

if __name__ == '__main__':

    shards = args.shards
    save_path = args.save_path
    meta_file = args.meta_file
    label_type = LabelType.EnergyForce

    logger.info(f'saving to {save_path}')

    futs = []
    with ProcessPoolExecutor(args.num_workers) as pool:
        for shard_id in range(shards):
            futs.append(
                pool.submit(MolDataset,
                            root=save_path,
                            meta_file=meta_file,
                            shard_id=shard_id,
                            shards=shards,
                            label_type=label_type,
                            max_nconfs=50,
                            process_only=True))
        wait(futs, return_when='FIRST_EXCEPTION')
        for fut in futs:
            if fut.exception() is not None:
                raise fut.exception()
