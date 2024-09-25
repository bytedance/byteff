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

import functools
import logging
from enum import IntEnum
from itertools import permutations

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from byteff.forcefield import CFF, get_gaff2_tfs, parse_tfs
from byteff.mol import Conformer, Molecule, MoleculeGraph, rkutil
from byteff.mol.moltools import find_equivalent_index, match_linear_proper
from byteff.utils.definitions import (MAX_RING_SIZE, MAX_TOTAL_CHARGE, BondOrder, ParamData, ParamTerm, TopoData,
                                      TopoParams, TopoTerm, atomic_number_map)

logger = logging.getLogger(__name__)


@functools.cache
def _cos_linspace(j, steps):
    x = torch.linspace(0, steps / 2, steps)
    return torch.cos(x * j) + x


def fake_coords(natoms, nconfs):
    coords = torch.stack([_cos_linspace(j, natoms) for j in range(3)]).T.unsqueeze(1)  # [natoms, 1, 3]
    coords = coords.repeat(1, nconfs, 1)  # [natoms, nconfs, 3]
    return coords


def get_ring_info(mol_graph: MoleculeGraph):
    ring_con, min_ring_size = [], []
    for atom in mol_graph.get_atoms():
        ring_con.append(atom.ring_connectivity)
        min_ring_size.append(atom.min_ring_size)
    return ring_con, min_ring_size


def get_tensor_params(topo_params: TopoParams, topo_term: TopoTerm, param_term: ParamTerm, index: list[list],
                      float_dtype):

    if len(index) == 0:
        return torch.zeros((0, ParamData.width[param_term]), dtype=float_dtype)
    topo = topo_params.topo[topo_term].tolist()
    if topo == index:
        params = topo_params[param_term].to(float_dtype)
        assert params.shape[0] == len(index)
        return params
    else:
        tp_map = {
            tuple(sorted(t)) if topo_term == TopoTerm.ImproperTorsion else tuple(t): p
            for t, p in zip(topo, topo_params.param[param_term].tolist())
        }
    params = []
    for atomidx in index:
        if topo_term == TopoTerm.ImproperTorsion:
            if tuple(sorted(atomidx)) in tp_map:
                params.append([tp_map[tuple(sorted(atomidx))][0]])
            else:
                params.append([0.])
        else:
            params.append(tp_map[tuple(atomidx)])
    params = torch.tensor(params, dtype=float_dtype)
    assert params.shape[0] == len(index)
    return params


class LabelType(IntEnum):
    Simple = 1  # using for write_itp
    Hessian = 2
    Torsion = 4
    GAFF2 = 8
    EnergyForce = 16
    ParitalHessianBase = 32
    ParitalHessian = 8 | 32


class MolData(Data):
    """ Molecule graph containing:
    name: str
    mapped_smiles: str
    x: IntTensor (atom_type, tot_charge, formal_charge, ring_con, min_ring_size)  # [natoms, 5]
    
    nAtom: IntTensor
    nBond: IntTensor
    nAngle: IntTensor
    nProperTorsion: IntTensor
    nImproperTorsion: IntTensor
    nNonbonded14: IntTensor
    nNonbondedAll: IntTensor if record_nonbonded_all

    edge_index: IntTensor  # [2, 2 * nBond]
    Atom_index: IntTensor  # [1, nAtom]
    Bond_index: IntTensor  # [2, nBond]
    Angle_index: IntTensor  # [3, nAngle]
    ProperTorsion_index: IntTensor  # [4, nProperTorsion]
    ImproperTorsion_index: IntTensor  # [4, nImproperTorsion]
    Nonbonded14_index: IntTensor  # [2, nNonbonded14]
    NonbondedAll_index: IntTensor  # [2, nNonbondedAll]  if record_nonbonded_all

    if add_bond_feature:
        bond_features: IntTensor (bond_ring, bond_order) # [nbonds, 2]
        Bond_edge_idx: IntTensor  # [nbonds, 1]
        Angle_edge_idx: IntTensor  # [nangles, 2]
        ProperTorsion_edge_idx: IntTensor  # [npropers, 3]
        ImproperTorsion_edge_idx: IntTensor  # [nimpropers, 3]

    ProperTorsion_linear_mask: FloatTensor  # [npropers]
    ProperTorsion_methyl_mask: FloatTensor  # [npropers]

    equi_index: IntTensor, symmetric atoms have the same index # [natoms]
    equi_edge_idx: IntTensor, symmetric edge have the same index  # [2 * nBond]

    if label_type & LabelType.GAFF2:
        Sigma_label: FloatTensor  # [natoms, 1]
        Epsilon_label: FloatTensor  # [natoms, 1]
        Charge_label: FloatTensor  # [natoms, 1]
        Bond_k_label: FloatTensor  # [nbonds, 1]
        Bond_length_label: FloatTensor  # [nbonds, 1]
        Angle_k_label: FloatTensor  # [nangles, 1]
        Angle_theta_label: FloatTensor  # [nangles, 1]
        ProperTorsion_k_label: FloatTensor  # [npropers, PROPERTORSION_TERMS]
        ImproperTorsion_k_label: FloatTensor  # [nimpropers, 1]

    if label_type & LabelType.TorsionBase:
        coords: FloatTensor   # [natoms, nconfs, 3]
        torsion_index: IntTensor  # [4, 1]
        confmask: FloatTensor  # [nconfs]
        energy_label: FloatTensor  # [batch_size, nconfs]
    
    if label_type & LabelType.HessianBase:
        coords: FloatTensor   # [natoms, 1, 3]
        Bond_k_label: FloatTensor, overwrite GAFF2  # [nbonds]
        Angle_k_label: FloatTensor, overwrite GAFF2  # [nangles]
        ImproperTorsion_k_label: FloatTensor, overwrite GAFF2  # [nimpropers]

    """

    @classmethod
    def from_mol(cls,
                 mol: Molecule,
                 label_type: LabelType,
                 partial_charges_key: str = "am1bcc_charges",
                 max_nconfs: int = 24,
                 record_nonbonded_all: bool = True,
                 add_bond_feature: bool = True,
                 nb_force_threshold: float = 0.,
                 int_dtype=torch.int32,
                 float_dtype=torch.float32) -> Data:
        """ Notice: attributes whose names contain 'index' are collated along dim -1, others along 0. """

        data = cls()
        data.name = mol.name
        data.mapped_smiles = mol.get_mapped_smiles()
        graph = MoleculeGraph(mol, max_include_ring=MAX_RING_SIZE)
        topos = graph.get_intra_topo()

        # atom features
        # atom_type, 0 - 9 embedding
        atom_type = torch.tensor([atomic_number_map[i] for i in mol.atomic_numbers], dtype=int_dtype)
        # total charge
        tot_charge = round(sum(mol.formal_charges))
        if not -MAX_TOTAL_CHARGE <= tot_charge <= MAX_TOTAL_CHARGE:
            logger.warning(f'total charge too large: {tot_charge}, mol name: {mol.name}')
        tot_charge_vec = torch.tensor([tot_charge] * mol.natoms, dtype=int_dtype)
        formal_charge_vec = torch.tensor(mol.formal_charges, dtype=int_dtype)
        # ring info
        ring_con, min_ring_size = get_ring_info(graph)
        ring_con = torch.tensor(ring_con, dtype=int_dtype)
        min_ring_size = torch.tensor(min_ring_size, dtype=int_dtype)

        features = torch.vstack([atom_type, tot_charge_vec, formal_charge_vec, ring_con, min_ring_size]).T
        data.x = features  # [natoms, 5]

        # edge_idx
        if add_bond_feature:
            bond_orders = list(BondOrder)
            edge_idx_dict = {}
            bond_features = []
            for i, atomidx in enumerate(topos['Bond']):
                edge_idx_dict[atomidx] = i
                edge_idx_dict[atomidx[::-1]] = i + len(topos['Bond'])
                bond = graph.get_bond(*atomidx)
                bond_features.append((int(bond.in_ring), bond_orders.index(BondOrder(bond.order))))
            data.edge_features = torch.tensor(bond_features + bond_features, dtype=int_dtype)

        # bonded terms
        for term, num in TopoData.width.items():
            if term in [TopoTerm.Nonbonded14, TopoTerm.NonbondedAll]:
                continue
            elif term is TopoTerm.Atom:
                atomidxs = [[i] for i in range(mol.natoms)]
            else:
                atomidxs = topos[term.name]
            index = torch.tensor(atomidxs, dtype=int_dtype).reshape(-1, num).T
            setattr(data, term.name + "_index", index)
            setattr(data, "n" + term.name, len(atomidxs))
            if add_bond_feature and term is not TopoTerm.Atom:
                edge_indices = []
                for ids in atomidxs:
                    eids = []
                    for i in range(len(ids) - 1):
                        if term != TopoTerm.ImproperTorsion:
                            eid = edge_idx_dict[(ids[i], ids[i + 1])]
                        else:
                            eid = edge_idx_dict[((ids[0], ids[i + 1]))]
                        eids.append(eid)
                    edge_indices.append(eids)
                setattr(data, term.name + "_edge_idx", torch.tensor(edge_indices, dtype=int_dtype).reshape(-1, num - 1))

        data.edge_index = torch.hstack((data.Bond_index, torch.flip(data.Bond_index, dims=(0,))))

        # nonbonded
        if record_nonbonded_all:
            nb14, nball = graph.get_nonbonded_pairs()
            data.Nonbonded14_index = torch.tensor(nb14, dtype=int_dtype).reshape(-1, 2).T  # [2, n_nb14]
            data.NonbondedAll_index = torch.tensor(nball, dtype=int_dtype).reshape(-1, 2).T  # [2, n_nball]
            data.nNonbonded14 = len(nb14)
            data.nNonbondedAll = len(nball)
        else:
            # skip calc NonbondedAll pairs to accelerate large molecule
            nb14 = graph.get_nonbonded14_pairs()
            data.Nonbonded14_index = torch.tensor(nb14, dtype=int_dtype).reshape(-1, 2).T  # [2, n_nb14]
            data.nNonbonded14 = len(nb14)

        # patch linear proper
        matches = match_linear_proper(mol)
        mask = torch.ones(data.nProperTorsion, dtype=float_dtype)
        for i, atomidx in enumerate(data.ProperTorsion_index.T):
            at = tuple(atomidx.tolist())
            if at in matches:
                mask[i] = 0.
        data.ProperTorsion_linear_mask = mask

        # record equivalent atom and bond
        atom_equi_index, bond_equi_index = find_equivalent_index(mol, data.edge_index.T.tolist())
        data.equi_index = torch.tensor(atom_equi_index, dtype=int_dtype)
        data.equi_edge_idx = torch.tensor(bond_equi_index, dtype=int_dtype)

        topo_params = None

        if label_type & LabelType.GAFF2:
            topo_params = parse_tfs(get_gaff2_tfs(mol), record_nonbonded_all=False)
            topo_params[ParamTerm.Charge] = [
                [c] for c in mol.get_partial_charges(partial_charges_key=partial_charges_key)
            ]

            for topo_term, param_terms in topo_params.topo_param_map.items():
                index = getattr(data, topo_term.name +"_index").T.tolist() \
                        if topo_term != TopoTerm.Atom else [[idx] for idx in range(mol.natoms)]
                for param_term in param_terms:
                    params = get_tensor_params(topo_params, topo_term, param_term, index, float_dtype=float_dtype)
                    setattr(data, param_term.name + '_label', params)

        if label_type & LabelType.ParitalHessianBase:
            coords = mol.get_confdata('coords')
            coords_t = torch.tensor(np.array(coords), dtype=float_dtype).transpose(0, 1)  # [natoms, nconfs, 3]
            data.coords = coords_t  # [natoms, nconfs, 3]

            hessian = mol.get_confdata('hessian')
            hessian = torch.tensor(np.array(hessian), dtype=float_dtype)  # [nconfs, natoms*3, natoms*3]
            nconfs = hessian.shape[0]

            partial_hessians = []
            ids_hessian_map = {}

            for term in [TopoTerm.Bond, TopoTerm.Angle, TopoTerm.ProperTorsion, TopoTerm.ImproperTorsion]:
                width = TopoData.width[term]
                rec = [[[] for _ in range(width)] for _ in range(width)]
                choices = list(permutations(range(width), 2))
                indices = getattr(data, f'{term.name}_index').T.tolist()
                for ids in indices:
                    for i, j in choices:
                        if abs(i - j) >= 3 and term is TopoTerm.ProperTorsion:
                            # In 4-members rings, 1-4 interaction of proper torsions should be taken into consideration.
                            # We omit this term for simplicity, which could result in an error of ~1 kcal/mol/A^2.
                            continue
                        a0, a1 = ids[i], ids[j]
                        if (a0, a1) not in ids_hessian_map:
                            ids_hessian_map[(a0, a1)] = len(partial_hessians)
                            partial_hessians.append(hessian[:, a0 * 3:a0 * 3 + 3,
                                                            a1 * 3:a1 * 3 + 3].reshape(nconfs, -1))
                        rec[i][j].append(ids_hessian_map[(a0, a1)])
                for i in range(width):
                    for j in range(width):
                        if i == j:
                            continue
                        if abs(i - j) >= 3 and term is TopoTerm.ProperTorsion:
                            continue
                        rec_ij = torch.tensor(rec[i][j], dtype=int_dtype)
                        assert len(rec_ij) == getattr(
                            data, f'n{term.name}'), f"{len(rec_ij)} != {getattr(data, f'n{term.name}')}"
                        setattr(data, f'{term.name}_rec_{i}_{j}', rec_ij)

            partial_hessians = torch.concat([p.unsqueeze(0) for p in partial_hessians],
                                            dim=0)  # [nPartialHessian, nconfs, 9]
            data.PartialHessian = partial_hessians
            data.nPartialHessian = partial_hessians.shape[0]

        if label_type & LabelType.Torsion:

            torsion_ids = mol.moledata['torsion_ids']
            data.torsion_index = torch.tensor(torsion_ids, dtype=int_dtype).unsqueeze(-1)  # [4, 1]

            label_energy = np.array(mol.get_confdata("energy"))
            label_energy -= label_energy.mean()
            label_energy = torch.tensor(label_energy, dtype=float_dtype)
            coords = mol.get_confdata('coords')
            coords = torch.tensor(np.array(coords), dtype=float_dtype).transpose(0, 1)  # [natoms, nconfs, 3]
            natoms, nconfs = coords.shape[0], coords.shape[1]
            confmask = torch.ones(max_nconfs)  # [torsion_nconfs]
            if nconfs >= max_nconfs:
                idx = torch.randperm(nconfs, dtype=torch.int64)
                coords = coords[:, idx[:max_nconfs]]
                label_energy = label_energy[idx[:max_nconfs]]
            else:
                # padding confs
                confmask[nconfs:] = 0.
                pad_coords = fake_coords(natoms, max_nconfs - nconfs)
                coords = torch.concat([coords, pad_coords], dim=1)  # [natoms, torsion_nconfs, 3]
                label_energy = torch.concat([label_energy, torch.zeros(max_nconfs - nconfs, dtype=float)], dim=0)
            data.confmask = confmask.unsqueeze(0)
            data.coords = coords
            data.label_energy = label_energy.reshape(1, -1)

        if label_type & LabelType.EnergyForce:

            label_energy = np.array(mol.get_confdata("energy"))
            label_energy -= label_energy.mean()
            label_energy = torch.tensor(label_energy, dtype=float_dtype)
            coords = mol.get_confdata('coords')
            coords = torch.tensor(np.array(coords), dtype=float_dtype).transpose(0, 1)  # [natoms, nconfs, 3]
            forces = mol.get_confdata('force')
            forces = torch.tensor(np.array(forces), dtype=float_dtype).transpose(0, 1)  # [natoms, nconfs, 3]
            natoms, nconfs = coords.shape[0], coords.shape[1]
            confmask = torch.ones(max_nconfs)  # [torsion_nconfs]
            if nconfs >= max_nconfs:
                idx = torch.randperm(nconfs, dtype=torch.int64)
                coords = coords[:, idx[:max_nconfs]]
                forces = forces[:, idx[:max_nconfs]]
                label_energy = label_energy[idx[:max_nconfs]]
            else:
                # padding confs
                confmask[nconfs:] = 0.
                pad_coords = fake_coords(natoms, max_nconfs - nconfs)
                coords = torch.concat([coords, pad_coords], dim=1)  # [natoms, torsion_nconfs, 3]
                forces = torch.concat([forces, torch.zeros_like(pad_coords)], dim=1)
                label_energy = torch.concat([label_energy, torch.zeros(max_nconfs - nconfs, dtype=float)], dim=0)
            data.confmask = confmask.unsqueeze(0)
            data.coords = coords
            data.forces = forces
            data.label_energy = label_energy.reshape(1, -1)

        # filter nonbonded force
        if hasattr(data, 'coords') and nb_force_threshold > 0.:
            if topo_params is None:
                topo_params = parse_tfs(get_gaff2_tfs(mol), record_nonbonded_all=False)
            topo_params[TopoTerm.Nonbonded14] = data.Nonbonded14_index.T
            topo_params[TopoTerm.NonbondedAll] = data.NonbondedAll_index.T
            topo_params[ParamTerm.Charge] = [
                [c] for c in mol.get_partial_charges(partial_charges_key=partial_charges_key)
            ]
            forces = CFF.calc_nonbonded14(topo_params, data.coords)['Nonbonded14_forces']
            forces += CFF.calc_nonbondedall(topo_params, data.coords)['NonbondedAll_forces']
            if not hasattr(data, "confmask"):
                if (forces.abs() > nb_force_threshold).any():
                    return None
            else:
                judge = torch.any(torch.any(forces.abs() > nb_force_threshold, dim=-1), dim=0)
                data.confmask[:, judge] = 0.
                if torch.sum(data.confmask) < 5.:
                    return None
        return data


def moldata_to_mols(graph: MolData) -> list[Molecule]:

    def create_mol(name, mapped_smiles, coords):
        nconfs = coords.shape[0]
        mol = Molecule.from_mapped_smiles(mapped_smiles, name=name)
        mol.append_conformers(Conformer(np.zeros((mol.natoms, 3)), mol.atomic_symbols))
        mol.conformers[0].coords = coords[0]
        if nconfs > 1:
            for i in range(1, nconfs):
                conformer = mol.conformers[0].copy()
                conformer.coords = coords[i]
                mol._conformers.append(conformer)
        return mol

    mols = []
    if hasattr(graph, 'batch') and graph.batch is not None:
        sum_natoms = 0
        coords = graph.coords.transpose(0, 1)
        for i, mapped_smiles in enumerate(graph.mapped_smiles):
            natoms = graph.natoms[i]
            m_coords = coords[:, sum_natoms:sum_natoms + natoms].detach().numpy()
            if hasattr(graph, 'confmask'):
                m_coords = m_coords[:round(torch.sum(graph.confmask[i]).item())]
            mols.append(create_mol(graph.name[i], mapped_smiles, m_coords))
            sum_natoms += natoms
    else:
        coords = graph.coords.transpose(0, 1).detach().numpy()
        if hasattr(graph, 'confmask'):
            coords = coords[:round(torch.sum(graph.confmask).item())]
        mols.append(create_mol(graph.name, graph.mapped_smiles, coords))
    return mols


def batch_single_moldata(data: MolData):
    loader = DataLoader([data])
    return next(iter(loader))


def batched_moldata_to_single(data: MolData, imol: int) -> MolData:
    moldata = MolData()

    atom_index = torch.cumsum(data.nAtom, dim=0)
    begin_aidx = atom_index[imol - 1] if imol > 0 else 0
    end_aidx = atom_index[imol]
    moldata.x = data.x[begin_aidx:end_aidx]
    moldata.coords = data.coords[begin_aidx:end_aidx]

    for k, v in data._store._mapping.items():

        if isinstance(v, list):
            setattr(moldata, k, v[imol])

    count_dict = {}
    for tt in TopoTerm:
        if f'n{tt.name}' in data:
            num_t = getattr(data, f'n{tt.name}')
            setattr(moldata, f'n{tt.name}', num_t[imol])
            num_t = torch.cumsum(num_t, dim=0)
            begin_tidx = num_t[imol - 1] if imol > 0 else 0
            end_tidx = num_t[imol]
            count_dict[tt] = (begin_tidx, end_tidx)
            index_name = f'{tt.name}_index'
            setattr(moldata, index_name, getattr(data, index_name)[:, begin_tidx:end_tidx] - begin_aidx)

            if tt in TopoParams.topo_param_map:
                for p in TopoParams.topo_param_map[tt]:
                    p_name = f'{p.name}_label'
                    if hasattr(data, p_name):
                        setattr(moldata, p_name, getattr(data, p_name)[begin_tidx:end_tidx])

    begin_eidx, end_eidx = 2 * count_dict[TopoTerm.Bond][0], 2 * count_dict[TopoTerm.Bond][1]
    moldata.edge_index = data.edge_index[:, begin_eidx:end_eidx] - begin_aidx
    moldata.edge_features = data.edge_features[begin_eidx:end_eidx]
    for k, v in count_dict.items():
        ename = f'{k.name}_edge_idx'
        if hasattr(data, ename):
            setattr(moldata, ename, getattr(data, ename)[v[0]:v[1]])

    moldata.equi_index = data.equi_index[begin_aidx:end_aidx] - begin_aidx
    moldata.equi_edge_idx = data.equi_edge_idx[begin_eidx:end_eidx]

    moldata.ProperTorsion_linear_mask = data.ProperTorsion_linear_mask[
        count_dict[TopoTerm.ProperTorsion][0]:count_dict[TopoTerm.ProperTorsion][1]]
    return moldata


def preprocess_mol(mol: Molecule, use_canonical_resonance=True):
    if use_canonical_resonance:
        conformers = mol.conformers
        rkmol = rkutil.get_canonical_resoner(mol.get_rkmol())
        mol = Molecule.from_rdkit(rkmol)
        mol._conformers = conformers

    label_type = LabelType.Simple
    graph = MolData.from_mol(mol, label_type, record_nonbonded_all=False)
    graph = batch_single_moldata(graph)
    return graph
