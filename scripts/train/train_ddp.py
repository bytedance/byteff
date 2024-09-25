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
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from byteff.train.trainer import ParamsTrainer
from byteff.utils import setup_default_logging

logger = setup_default_logging()

parser = argparse.ArgumentParser(description='train ddp')
parser.add_argument('--conf', type=str, default='config.yaml')
parser.add_argument('--partial_train', type=bool, default=False, action=argparse.BooleanOptionalAction)
parser.add_argument('--timestamp', type=bool, default=False, action=argparse.BooleanOptionalAction)
parser.add_argument('--restart', type=bool, default=False, action=argparse.BooleanOptionalAction)
args = parser.parse_args()


def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)


def ddp_cleanup():
    dist.destroy_process_group()


def ddp_train(local_rank: int, world_size: int, trainer: ParamsTrainer):
    if "MLP_ROLE_INDEX" in os.environ:
        rank = local_rank + int(os.environ["MLP_WORKER_GPU"]) * int(os.environ["MLP_ROLE_INDEX"])
    else:
        rank = local_rank

    device = torch.device('cuda', local_rank)
    ddp_setup(rank, world_size)

    trainer.start_ddp(rank, world_size, device, find_unused_parameters=args.partial_train, restart=args.restart)
    trainer.train_loop()

    ddp_cleanup()


def main():

    assert os.path.exists(args.conf) and args.conf.endswith('.yaml'), f'yaml config {args.conf} not found.'

    local_size = world_size = torch.cuda.device_count()
    master_addr = "localhost"
    master_port = "12355"
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port

    trainer = ParamsTrainer(args.conf, ddp=True, timestamp=args.timestamp, restart=args.restart)

    try:
        mp.start_processes(ddp_train, args=(world_size, trainer), nprocs=local_size, join=True, daemon=False)
        trainer.logger.info("Training finished!")
    except Exception as e:
        trainer.logger.exception('train failed: %s', e)
        raise e


if __name__ == "__main__":
    main()
