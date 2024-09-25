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

from byteff.train.trainer import ParamsTrainer
from byteff.utils import setup_default_logging
from byteff.utils.definitions import ParamTerm

logger = setup_default_logging()

parser = argparse.ArgumentParser(description='train local')
parser.add_argument('--conf', type=str, default='config.yaml')
parser.add_argument('--timestamp', type=bool, default=False, action=argparse.BooleanOptionalAction)
parser.add_argument('--restart', type=bool, default=False, action=argparse.BooleanOptionalAction)
parser.add_argument('--only_bond_angle', type=bool, default=False, action=argparse.BooleanOptionalAction)
parser.add_argument('--only_bonded', type=bool, default=False, action=argparse.BooleanOptionalAction)
args = parser.parse_args()


def main():

    assert os.path.exists(args.conf) and args.conf.endswith('.yaml'), f'yaml config {args.conf} not found.'

    trainer = ParamsTrainer(args.conf, timestamp=args.timestamp, restart=args.restart)

    if args.only_bond_angle or args.only_bonded:
        # freeze all parameters
        for param in trainer.model.parameters():
            param.requires_grad = False

        if args.only_bond_angle:
            trainable_set = {
                ParamTerm.Bond_k.name, ParamTerm.Bond_length.name, ParamTerm.Angle_k.name, ParamTerm.Angle_theta.name
            }
        elif args.only_bonded:
            trainable_set = {
                ParamTerm.Bond_k.name,
                ParamTerm.Bond_length.name,
                ParamTerm.Angle_k.name,
                ParamTerm.Angle_theta.name,
                ParamTerm.ProperTorsion_k.name,
                ParamTerm.ImproperTorsion_k.name,
            }

        # enable output k
        for name, param in trainer.model.output_layer.named_parameters():
            if name.split('.')[1] in trainable_set:
                param.requires_grad = True

    trainer.train_loop()
    logger.info("Training finished!")


if __name__ == "__main__":
    main()
