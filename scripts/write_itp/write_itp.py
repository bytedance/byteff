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

from byteff.data import preprocess_mol
from byteff.forcefield import topo_params_to_tfs
from byteff.mol import Molecule
from byteff.train import ParamsTrainer

parser = argparse.ArgumentParser(description='label a molecule with a trained model')
parser.add_argument('--SMILES', type=str, required=True, help='SMILES of a molecule')
parser.add_argument('--save_name', type=str, required=True, help='the path to save resulting itp file')
parser.add_argument('--train_path', type=str, required=True, help='the working directory of training')
args = parser.parse_args()

if __name__ == '__main__':

    config_path = os.path.join(args.train_path, "fftrainer_config_in_use.yaml")
    trainer = ParamsTrainer(config_path, device='cpu', load_data=False, make_working_dir=False, timestamp=False)
    trainer.load_ckpt(trainer.config.optimal_path())
    model = trainer.model
    model.eval()

    mol = Molecule.from_smiles(args.SMILES)
    graph = preprocess_mol(mol)
    topo_params = model(graph)
    tfs = topo_params_to_tfs(topo_params, mol)
    tfs.write_itp(args.save_name)