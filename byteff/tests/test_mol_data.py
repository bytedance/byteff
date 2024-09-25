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
import os

from byteff.data import LabelType, MolData
from byteff.mol import Molecule, MolTarLoader
from byteff.utils import get_data_file_path
from byteff.utils.definitions import atomic_number_map

logger = logging.getLogger(__name__)


def test_mol_data_gaff2():

    meta_file = get_data_file_path("dataset_hessian/meta.txt", "byteff.tests.testdata")
    loader = MolTarLoader(os.path.dirname(meta_file), os.path.basename(meta_file))

    for mol in loader[:20]:

        logger.info(mol.get_mapped_smiles())

        data = MolData.from_mol(mol, LabelType.GAFF2, partial_charges_key="am1bcc_charges")
        assert data.name == mol.name
        assert data.x[:, 0].tolist() == [atomic_number_map[i] for i in mol.atomic_numbers]
        tot_charge = sum(mol.formal_charges)
        assert data.x[:, 1].tolist() == [tot_charge for _ in mol.formal_charges]
        assert data.x[:, 2].tolist() == mol.formal_charges

        assert data.Bond_index.shape[1] == data.Bond_k_label.shape[0] == data.Bond_length_label.shape[0]
        assert data.Angle_index.shape[1] == data.Angle_k_label.shape[0] == data.Angle_theta_label.shape[0]
        assert data.ProperTorsion_index.shape[1] == data.ProperTorsion_k_label.shape[0]
        assert data.ImproperTorsion_index.shape[1] == data.ImproperTorsion_k_label.shape[0]

        assert data.edge_features.shape == (data.Bond_index.shape[1] * 2, 2)
        assert data.Bond_edge_idx.shape[0] == data.Bond_index.shape[1]
        assert data.Angle_edge_idx.shape[0] == data.Angle_index.shape[1]
        assert data.ProperTorsion_edge_idx.shape[0] == data.ProperTorsion_index.shape[1]
        assert data.ImproperTorsion_edge_idx.shape[0] == data.ImproperTorsion_index.shape[1]

        assert data.Bond_index.shape[0] - 1 == data.Bond_edge_idx.shape[1] == 2 - 1
        assert data.Angle_index.shape[0] - 1 == data.Angle_edge_idx.shape[1] == 3 - 1
        assert data.ProperTorsion_index.shape[0] - 1 == data.ProperTorsion_edge_idx.shape[1] == 4 - 1
        assert data.ImproperTorsion_index.shape[0] - 1 == data.ImproperTorsion_edge_idx.shape[1] == 4 - 1

        assert list(data.Charge_label.shape) == [mol.natoms, 1]
        assert list(data.Sigma_label.shape) == [mol.natoms, 1]
        assert list(data.Epsilon_label.shape) == [mol.natoms, 1]

        assert list(data.Nonbonded14_index.shape) == [2, data.nNonbonded14]
        assert list(data.NonbondedAll_index.shape) == [2, data.nNonbondedAll]


def test_mol_data_partial_hessian():

    meta_file = get_data_file_path("dataset_hessian/meta.txt", "byteff.tests.testdata")
    loader = MolTarLoader(os.path.dirname(meta_file), os.path.basename(meta_file))

    for mol in loader[20:40]:

        logger.info(mol.get_mapped_smiles())

        data = MolData.from_mol(mol, LabelType.ParitalHessianBase, partial_charges_key="am1bcc_charges")
        assert data.name == mol.name
        assert data.x[:, 0].tolist() == [atomic_number_map[i] for i in mol.atomic_numbers]
        tot_charge = sum(mol.formal_charges)
        assert data.x[:, 1].tolist() == [tot_charge for _ in mol.formal_charges]
        assert data.x[:, 2].tolist() == mol.formal_charges

        pairs = set()
        for i, j in data.Bond_index.T.tolist():
            pairs.add(tuple(sorted([i, j])))
        for i, j, k in data.Angle_index.T.tolist():
            pairs.add(tuple(sorted([i, k])))
        assert data.PartialHessian.shape[0] == 2 * len(pairs) == data.nPartialHessian

        assert data.edge_features.shape == (data.Bond_index.shape[1] * 2, 2)
        assert data.Bond_edge_idx.shape[0] == data.Bond_index.shape[1]
        assert data.Angle_edge_idx.shape[0] == data.Angle_index.shape[1]
        assert data.ProperTorsion_edge_idx.shape[0] == data.ProperTorsion_index.shape[1]
        assert data.ImproperTorsion_edge_idx.shape[0] == data.ImproperTorsion_index.shape[1]

        assert data.Bond_index.shape[0] - 1 == data.Bond_edge_idx.shape[1] == 2 - 1
        assert data.Angle_index.shape[0] - 1 == data.Angle_edge_idx.shape[1] == 3 - 1
        assert data.ProperTorsion_index.shape[0] - 1 == data.ProperTorsion_edge_idx.shape[1] == 4 - 1
        assert data.ImproperTorsion_index.shape[0] - 1 == data.ImproperTorsion_edge_idx.shape[1] == 4 - 1

        assert list(data.Nonbonded14_index.shape) == [2, data.nNonbonded14]
        assert list(data.NonbondedAll_index.shape) == [2, data.nNonbondedAll]
        assert list(data.coords.shape) == [mol.natoms, 1, 3]


def test_mol_data_torsion():

    meta_file = get_data_file_path("dataset_torsion/meta.txt", "byteff.tests.testdata")
    loader = MolTarLoader(os.path.dirname(meta_file), os.path.basename(meta_file))

    nconfs = 10
    for mol in loader[:10]:
        logger.info(mol.get_mapped_smiles())
        data = MolData.from_mol(mol, LabelType.Torsion, partial_charges_key="am1bcc_charges", max_nconfs=nconfs)

        assert list(data.coords.shape) == [mol.natoms, nconfs, 3]
        assert data.confmask.tolist() == [[1.] * nconfs]

    nconfs = 30
    for mol in loader[10:20]:
        logger.info(mol.get_mapped_smiles())
        data = MolData.from_mol(mol,
                                LabelType.Torsion | LabelType.GAFF2,
                                partial_charges_key="am1bcc_charges",
                                max_nconfs=nconfs)

        assert list(data.coords.shape) == [mol.natoms, nconfs, 3]
        assert data.confmask.tolist() == [[1.] * mol.nconfs + [0.] * (nconfs - mol.nconfs)]


def test_linear_proper():

    mol = Molecule.from_mapped_smiles(
        "[C:1]([c:2]1[c:3]([H:19])[c:4]2[c:5]([c:6]([H:20])[c:7]1[C:8]([H:21])"
        "([H:22])[H:23])[C:9](=[O:10])[C-:11]([C:12]#[N:13])[C:14]2=[O:15])([H:16])([H:17])[H:18]",
        nconfs=1)
    mol.set_partial_charges([0.] * mol.natoms)

    data = MolData.from_mol(mol, LabelType.Simple)
    assert round(data.ProperTorsion_linear_mask.sum().item()) == data.nProperTorsion - 2


def test_nbforce_threshold():
    mol = Molecule(get_data_file_path('torsion_example.xyz', 'byteff.tests.testdata'))
    data = MolData.from_mol(mol, LabelType.Torsion, partial_charges_key="am1bcc_charges", nb_force_threshold=100.)
    assert data.confmask.sum() == 24.

    mol = Molecule(get_data_file_path('torsion_bad.xyz', 'byteff.tests.testdata'))
    data = MolData.from_mol(mol, LabelType.Torsion, partial_charges_key="am1bcc_charges", nb_force_threshold=100.)
    assert data.confmask.sum() == 18.
