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
from datetime import datetime

import numpy as np
import torch
from torch.optim import Adam

from byteff.forcefield import CFF, ff_kernels
from byteff.forcefield.topparse import (AngleTypeEnum, BondTypeEnum, DihedralTypeEnum, LJCombinationRuleEnum,
                                        NonbondedFunctionEnum, PairTypeEnum, RecordAngle, RecordAtom, RecordAtomType,
                                        RecordBond, RecordDihedral, RecordMoleculeType, RecordPair, Records,
                                        RecordSection, RecordText, TopoAtomTypes, TopoDefaults, TopoFullSystem)
from byteff.mol import Molecule, MoleculeGraph, Topology
from byteff.mol.rkutil import sorted_atomids
from byteff.utils import run_command_and_check
from byteff.utils import simple_unit as unit
from byteff.utils import temporary_cd
from byteff.utils.definitions import (PROPERTORSION_TERMS, ParamTerm, TopoParams, TopoTerm)

logger = logging.getLogger(__name__)


def get_gaff2_tfs(mol: Molecule, conf_id: int = 0) -> TopoFullSystem:

    mol_name = 'MOL'
    with temporary_cd():
        # Molecule to .sdf
        sdf_path = mol_name + ".sdf"
        mol.to_sdf(sdf_path, conf_id=conf_id)

        # .sdf to .mol2
        mol2_path = mol_name + '.mol2'
        run_command_and_check(f'obabel {sdf_path} -O {mol2_path} -xl', allow_error=False, separate_stderr=True)

        # .mol2 to .itp
        total_formal_charges = sum(mol.formal_charges)
        run_command_and_check(
            f'acpype -i {mol2_path} -c user -a gaff2 -o gmx -b {mol_name} -f -n {total_formal_charges}',
            allow_error=False,
            separate_stderr=True)

        return TopoFullSystem.from_file(f'{mol_name}.acpype/{mol_name}_GMX.itp')


def parse_tfs(tfs: TopoFullSystem, record_nonbonded_all=True) -> TopoParams:
    topo_params = TopoParams()
    itp_mol = tfs.mol_topos[0]

    # atom types
    vdw_params = {}
    atom_types = TopoAtomTypes(tfs.uuid)
    for atomtype in atom_types.atomtypes:
        sigma, epsilon = unit.nm_to_A(atomtype.sigma), unit.kj_to_kcal(atomtype.epsilon)
        vdw_params[atomtype.name] = (sigma, epsilon)

    # atoms
    atom_ids, sigma_list, epsilon_list, charge_list = [], [], [], []
    for idx, atom in enumerate(itp_mol.atoms):
        assert idx == atom.nr - 1
        atom_ids.append([idx])
        sigma_list.append([vdw_params[atom.atype][0]])
        epsilon_list.append([vdw_params[atom.atype][1]])
        charge_list.append([atom.charge])
    topo_params[TopoTerm.Atom] = atom_ids
    topo_params[ParamTerm.Sigma] = sigma_list
    topo_params[ParamTerm.Epsilon] = epsilon_list
    topo_params[ParamTerm.Charge] = charge_list

    # bonds
    bond_ids, ks, lengths = [], [], []
    for bond in itp_mol.bonds:
        bond_ids.append(sorted_atomids([bond.ai - 1, bond.aj - 1]))
        ks.append([unit.kj_mol_nm2_to_kcal_mol_A2(bond.c1)])
        lengths.append([unit.nm_to_A(bond.c0)])
    topo_params[TopoTerm.Bond] = bond_ids
    topo_params[ParamTerm.Bond_k] = ks
    topo_params[ParamTerm.Bond_length] = lengths

    # angles
    angle_ids, ks, thetas = [], [], []
    for angle in itp_mol.angles:
        angle_ids.append(sorted_atomids((angle.ai - 1, angle.aj - 1, angle.ak - 1)))
        ks.append([unit.kJ_mol_to_kcal_mol(angle.c1)])
        thetas.append([angle.c0])
    topo_params[TopoTerm.Angle] = angle_ids
    topo_params[ParamTerm.Angle_k] = ks
    topo_params[ParamTerm.Angle_theta] = thetas

    # torsions
    proper_ks_map = dict()
    improper_ids, improper_ks = [], []
    for dihedral in itp_mol.dihedrals:
        idxs = (dihedral.ai - 1, dihedral.aj - 1, dihedral.ak - 1, dihedral.al - 1)
        phase = dihedral.c0
        k = unit.kj_to_kcal(dihedral.c1)
        periodicity = round(dihedral.c2)
        assert abs(periodicity - dihedral.c2) < 1e-3, f'non-integer periodicity {dihedral.c2} detected.'
        if dihedral.funct == DihedralTypeEnum.MULTIPLE_PROPER:
            idxs = sorted_atomids(idxs)
            if abs(phase - periodicity % 2 * 180.) < 1e-3:
                k = -k
            else:
                assert abs(phase - (periodicity - 1) % 2 * 180.) < 1e-3, f'phase {phase} is not allowed.'
            if idxs not in proper_ks_map:
                proper_ks_map[idxs] = [0.] * PROPERTORSION_TERMS
            proper_ks_map[idxs][periodicity - 1] = k
        elif dihedral.funct == DihedralTypeEnum.PERIODIC_IMPROPER:
            assert periodicity == 2 and abs(phase - 180.) < 1e-3
            improper_ids.append(idxs)
            improper_ks.append([k])
        else:
            raise ValueError(f'unsupported funct {dihedral.funct} detected.')
    proper_ids, proper_ks = [], []
    for k, v in proper_ks_map.items():
        proper_ids.append(k)
        proper_ks.append(v)
    topo_params[TopoTerm.ProperTorsion] = proper_ids
    topo_params[TopoTerm.ImproperTorsion] = improper_ids
    topo_params[ParamTerm.ProperTorsion_k] = proper_ks
    topo_params[ParamTerm.ImproperTorsion_k] = improper_ks

    # nonbonded14
    itp_pairs = set()
    for pair in itp_mol.pairs:
        itp_pairs.add(sorted_atomids((pair.ai - 1, pair.aj - 1)))
    topo_params[TopoTerm.Nonbonded14] = list(itp_pairs)

    if record_nonbonded_all:
        moltopo = Topology(bonds=bond_ids)
        moltopo_nonbondedall = set(moltopo.nonbondedall_pairs)
        topo_params[TopoTerm.NonbondedAll] = list(moltopo_nonbondedall)

    return topo_params


def make_hessian_mask(topo_params: TopoParams):
    natoms = len(topo_params[TopoTerm.Atom])
    mask = torch.zeros((natoms, natoms), dtype=torch.float32)
    for term in topo_params[TopoTerm.Bond]:
        mask[term[0], term[-1]] = 1.
    for term in topo_params[TopoTerm.Angle]:
        mask[term[0], term[-1]] = 1.
    # for term in topo_params[TopoTerm.ProperTorsion]:
    #     mask[term[0], term[-1]] = 1.
    mask = mask.repeat_interleave(3, -1)
    mask = mask.repeat_interleave(3, -2)
    return mask


def hessian_masked_mape(label: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor):

    def sum_dirrections(in_tensor: torch.Tensor) -> torch.Tensor:
        na = in_tensor.shape[0] // 3
        in_tensor = in_tensor.reshape(na, 3, na, 3)
        in_tensor = in_tensor.sum(-1)
        in_tensor = in_tensor.sum(-2)
        return in_tensor

    diff = (pred - label) * mask
    loss = sum_dirrections(torch.abs(diff))
    div = sum_dirrections(torch.abs(label))
    loss /= torch.where(div < torch.finfo().eps, 1., div)
    return loss.sum() / mask.sum()


def calc_hessian_loss(tp, coords, mask, label):
    pred = CFF.energy_force(
        tp,
        coords,
        calc_hessian=True,
        calc_terms={TopoTerm.Bond, TopoTerm.Angle, TopoTerm.ProperTorsion, TopoTerm.ImproperTorsion})
    hessian = pred['hessian'][..., 0]
    hessian_loss = hessian_masked_mape(label, hessian, mask)
    return hessian_loss, pred['ImproperTorsion_energy']


def calc_force_loss(tp: TopoParams, coords):

    tp1 = tp.copy()
    for term in ParamTerm:
        if term is not ParamTerm.ImproperTorsion_k:
            tp1[term] = tp1[term].clone().detach()

    pred = CFF.energy_force(tp1, coords, calc_hessian=False, calc_force=True)
    force_loss = torch.mean(torch.square(pred['forces']))
    return force_loss


def topo_params_from_fitting(mol: Molecule,
                             partial_charges_key='am1bcc_charges',
                             conf_id: int = 0,
                             use_improper: bool = False,
                             record_nonbonded_all: bool = False) -> TopoParams:

    coords = torch.tensor(mol.conformers[conf_id].coords, dtype=torch.float32).unsqueeze(-2)
    label_hessian = torch.tensor(mol.conformers[conf_id].confdata['hessian'], dtype=torch.float32)
    tp = parse_tfs(get_gaff2_tfs(mol), record_nonbonded_all=record_nonbonded_all)
    tp[ParamTerm.Charge] = [[c] for c in mol.get_partial_charges(conf_id, partial_charges_key=partial_charges_key)]

    molgraph = MoleculeGraph(mol)
    bond_list = molgraph.get_bonds()
    angle_list = molgraph.get_angles()
    improper_list = molgraph.get_impropers()

    tp[TopoTerm.Bond] = bond_list
    tp[TopoTerm.Angle] = angle_list
    tp[ParamTerm.Bond_length] = ff_kernels.get_distance_vec(coords, tp[TopoTerm.Bond].long())[0]
    tp[ParamTerm.Angle_theta] = ff_kernels.get_angle_vec(coords, tp[TopoTerm.Angle].long(),
                                                         with_vec=False)[0] / torch.pi * 180.
    # use 800 and 80 as init
    tp[ParamTerm.Bond_k] = torch.ones((len(bond_list), 1), dtype=torch.float32) * 800
    tp[ParamTerm.Angle_k] = torch.ones((len(angle_list), 1), dtype=torch.float32) * 80
    bond_k_log = torch.log(tp[ParamTerm.Bond_k]).clone().detach()
    angle_k_log = torch.log(tp[ParamTerm.Angle_k]).clone().detach()
    bond_k_log.requires_grad = True
    angle_k_log.requires_grad = True
    parameters = [bond_k_log, angle_k_log]

    if use_improper:
        tp[TopoTerm.ImproperTorsion] = improper_list
        tp[ParamTerm.ImproperTorsion_k] = torch.ones((len(improper_list), 1), dtype=torch.float32)
        improper_k_log = torch.log(tp[ParamTerm.ImproperTorsion_k]).clone().detach()
        improper_k_log.requires_grad = True
        parameters.append(improper_k_log)
    else:
        tp[TopoTerm.ImproperTorsion] = []
        tp[ParamTerm.ImproperTorsion_k] = []

    optimizer = Adam(parameters, lr=.1)

    mask = make_hessian_mask(tp)

    losses = []
    early_stop_threshold = 0.001

    for i in range(1000):

        optimizer.zero_grad()
        tp[ParamTerm.Bond_k] = 100. + torch.exp(bond_k_log)
        tp[ParamTerm.Angle_k] = 40. + torch.exp(angle_k_log)

        if use_improper:
            tp[ParamTerm.ImproperTorsion_k] = torch.exp(improper_k_log)

        hessian_loss, energy_loss = calc_hessian_loss(tp, coords, mask, label_hessian)
        # force_loss = calc_force_loss(tp, coords)

        p_sum = tp[ParamTerm.Bond_k].sum() + tp[ParamTerm.Angle_k].sum() + tp[ParamTerm.ImproperTorsion_k].sum()
        counts = tp[ParamTerm.Bond_k].nelement() + tp[ParamTerm.Angle_k].nelement() + tp[
            ParamTerm.ImproperTorsion_k].nelement()
        l1 = p_sum / counts
        # loss = 100 * hessian_loss + 0.1 * force_loss + 0.001 * l1
        loss = 100 * hessian_loss + 0.1 * energy_loss + 0.001 * l1
        # loss = 100 * hessian_loss + 0.001 * l1
        losses.append(loss.item())
        if i > 100 and abs(losses[-1] - losses[-2]) < early_stop_threshold and abs(losses[-1] -
                                                                                   losses[-20]) < early_stop_threshold:
            break

        loss.backward()
        optimizer.step()

    if losses[-1] > losses[0]:
        raise RuntimeError(f'loss did not improved: {losses[0]}, {losses[-1]}')
    if tp[ParamTerm.Bond_k].isnan().any() or tp[ParamTerm.Angle_k].isnan().any() or tp[
            ParamTerm.ImproperTorsion_k].isnan().any():
        raise RuntimeError(f'found nan')

    logger.info(f'{mol.name}, loss: {losses[0]:.4f}, {losses[-1]:.4f}')

    tp[ParamTerm.Bond_k] = tp[ParamTerm.Bond_k].clone().detach()
    tp[ParamTerm.Angle_k] = tp[ParamTerm.Angle_k].clone().detach()
    tp[ParamTerm.ImproperTorsion_k] = tp[ParamTerm.ImproperTorsion_k].clone().detach()

    return tp


def topo_params_to_tfs(topo_params: TopoParams, mol: Molecule, mol_name='MOL'):

    uncertainty = topo_params.uncertainty
    records = Records()
    comment = f"; ITP file created by ByteFF-ML, {datetime.now()}"
    record = RecordText(text="", comment=comment)
    records.all.append(record)

    # atomtypes, atoms
    atomtypes = []
    atoms = []
    indices = topo_params[TopoTerm.Atom].flatten().tolist()
    sigma_list = topo_params[ParamTerm.Sigma].flatten().tolist()
    epsilon_list = topo_params[ParamTerm.Epsilon].flatten().tolist()
    charge_list = topo_params[ParamTerm.Charge].flatten().tolist()
    sigma_unc = uncertainty[ParamTerm.Sigma].flatten().tolist() if uncertainty[ParamTerm.Sigma] is not None else None
    epsilon_unc = uncertainty[ParamTerm.Epsilon].flatten().tolist() if uncertainty[
        ParamTerm.Epsilon] is not None else None
    charge_unc = uncertainty[ParamTerm.Charge].flatten().tolist() if uncertainty[ParamTerm.Charge] is not None else None
    for i, (atomidx, sigma, epsilon, charge) in enumerate(zip(indices, sigma_list, epsilon_list, charge_list)):
        atom = mol.rkmol.GetAtomWithIdx(atomidx)
        element = atom.GetSymbol()
        at_num = atom.GetAtomicNum()
        name = f"{element.lower()}{atomidx}bf"
        if sigma_unc is not None:
            comment1 = f"    ;  uncertainty sigma = {unit.A_to_nm(sigma_unc[i]):.6f},"\
                      f" epsilon = {unit.kcal_to_kJ(epsilon_unc[i]):.6f}"
            comment2 = f"    ;  uncertainty charge = {charge_unc[i]:.6f}"
        else:
            comment1, comment2 = None, None

        atom_type = RecordAtomType(name=name,
                                   at_num=at_num,
                                   V=unit.A_to_nm(sigma),
                                   W=unit.kcal_to_kJ(epsilon),
                                   comment=comment1)
        atomtypes.append(atom_type)

        mass = atom.GetMass()
        atom = RecordAtom(nr=atomidx + 1,
                          atype=name,
                          resnr=1,
                          residue="UNL",
                          atom=name[:-2],
                          cgnr=atomidx + 1,
                          charge=charge,
                          mass=mass,
                          comment=comment2)
        atoms.append(atom)

    records.all.append(RecordSection(section="atomtypes"))
    records.all += atomtypes
    records.all.append(RecordSection(section="moleculetype"))
    records.all.append(RecordMoleculeType(name=mol_name, nrexcl=3))
    records.all.append(RecordSection(section="atoms"))
    records.all += atoms

    # bonds
    indices = topo_params[TopoTerm.Bond].tolist()
    bond_k_list = topo_params[ParamTerm.Bond_k].flatten().tolist()
    bond_l_list = topo_params[ParamTerm.Bond_length].flatten().tolist()
    bond_k_unc = uncertainty[ParamTerm.Bond_k].flatten().tolist() if uncertainty[ParamTerm.Bond_k] is not None else None
    bond_l_unc = uncertainty[ParamTerm.Bond_length].flatten().tolist() if uncertainty[
        ParamTerm.Bond_length] is not None else None
    bonds = []
    for i, atomidx in enumerate(indices):
        if bond_k_unc is not None:
            comment = f"    ;  uncertainty r = {unit.A_to_nm(bond_l_unc[i]):.6f},"\
                      f" k = {unit.kcal_mol_A2_to_kJ_mol_nm2(bond_k_unc[i]):.6f}"
        else:
            comment = None
        bond = RecordBond(ai=atomidx[0] + 1,
                          aj=atomidx[1] + 1,
                          funct=BondTypeEnum.BOND,
                          c0=unit.A_to_nm(bond_l_list[i]),
                          c1=unit.kcal_mol_A2_to_kJ_mol_nm2(bond_k_list[i]),
                          comment=comment)
        bonds.append(bond)
    if bonds:
        records.all.append(RecordSection(section="bonds"))
        records.all += bonds

    # angles
    indices = topo_params[TopoTerm.Angle].tolist()
    angle_k_list = topo_params[ParamTerm.Angle_k].flatten().tolist()
    angle_t_list = topo_params[ParamTerm.Angle_theta].flatten().tolist()
    angle_k_unc = uncertainty[ParamTerm.Angle_k].flatten().tolist() if uncertainty[
        ParamTerm.Angle_k] is not None else None
    angle_t_unc = uncertainty[ParamTerm.Angle_theta].flatten().tolist() if uncertainty[
        ParamTerm.Angle_theta] is not None else None
    angles = []
    for i, atomidx in enumerate(indices):
        if angle_k_unc is not None:
            comment = f"    ;  uncertainty theta = {angle_t_unc[i]:.6f}, k = {unit.kcal_to_kJ(angle_k_unc[i]):.6f}"
        else:
            comment = None

        angle = RecordAngle(ai=atomidx[0] + 1,
                            aj=atomidx[1] + 1,
                            ak=atomidx[2] + 1,
                            funct=AngleTypeEnum.ANGLE,
                            c0=angle_t_list[i],
                            c1=unit.kcal_to_kJ(angle_k_list[i]),
                            comment=comment)
        angles.append(angle)
    if angles:
        records.all.append(RecordSection(section="angles"))
        records.all += angles

    # propers
    indices = topo_params[TopoTerm.ProperTorsion].tolist()
    proper_k_list = topo_params[ParamTerm.ProperTorsion_k].tolist()
    proper_k_unc = uncertainty[ParamTerm.ProperTorsion_k].tolist() if uncertainty[
        ParamTerm.ProperTorsion_k] is not None else None
    propers = []
    for i, atomidx in enumerate(indices):
        for ip, period in enumerate(range(PROPERTORSION_TERMS)):
            if proper_k_unc is not None:
                comment = f"    ;  uncertainty k = {unit.kcal_to_kJ(proper_k_unc[i][ip]):.6f}"
            else:
                comment = None
            record = RecordDihedral(ai=atomidx[0] + 1,
                                    aj=atomidx[1] + 1,
                                    ak=atomidx[2] + 1,
                                    al=atomidx[3] + 1,
                                    funct=DihedralTypeEnum.MULTIPLE_PROPER,
                                    c0=(period % 2) * 180.,
                                    c1=unit.kcal_to_kJ(proper_k_list[i][ip]),
                                    c2=period + 1,
                                    comment=comment)
            propers.append(record)
    if propers:
        records.all.append(RecordSection(section="dihedrals"))
        records.all += propers

    # impropers
    indices = topo_params[TopoTerm.ImproperTorsion].tolist()
    improper_k_list = topo_params[ParamTerm.ImproperTorsion_k].flatten().tolist()
    improper_k_unc = uncertainty[ParamTerm.ImproperTorsion_k].flatten().tolist() if uncertainty[
        ParamTerm.ImproperTorsion_k] is not None else None
    impropers = []
    seqs = [[0, 1, 2], [1, 2, 0], [2, 0, 1]]
    for i, atomidx in enumerate(indices):
        for s in seqs:
            ijkl = (atomidx[0], atomidx[1 + s[0]], atomidx[1 + s[1]], atomidx[1 + s[2]])
            if improper_k_unc is not None:
                comment = f"    ;  uncertainty k = {unit.kcal_to_kJ(improper_k_unc[i])/3:.6f}"
            else:
                comment = None
            if improper_k_list[i] > 1e-4:
                record = RecordDihedral(ai=ijkl[0] + 1,
                                        aj=ijkl[1] + 1,
                                        ak=ijkl[2] + 1,
                                        al=ijkl[3] + 1,
                                        funct=DihedralTypeEnum.PERIODIC_IMPROPER,
                                        c0=180.,
                                        c1=unit.kcal_to_kJ(improper_k_list[i]) / 3,
                                        c2=2,
                                        comment=comment)
                impropers.append(record)
    if impropers:
        records.all.append(RecordSection(section="dihedrals"))
        records.all += impropers

    # pairs
    pairs = set()
    pair_list = []
    for atomidx in topo_params[TopoTerm.Nonbonded14].tolist():
        pair = tuple(atomidx)
        if pair not in pairs:
            pairs.add(pair)
            record = RecordPair(ai=pair[0] + 1, aj=pair[1] + 1, funct=PairTypeEnum.EXTRA_LJ)
            pair_list.append(record)
    if pair_list:
        records.all.append(RecordSection(section="pairs"))
        records.all += pair_list

    tfs = TopoFullSystem.from_records(records=records.all, sort_idx=True, round_on="w")
    # amber style [ defaults ]
    td = TopoDefaults(tfs.uuid)
    assert td.nbfunc == NonbondedFunctionEnum.LENNARD_JONES
    assert td.comb_rule == LJCombinationRuleEnum.SIGMA_EPSILON
    assert td.gen_pairs == "yes"
    assert np.isclose(td.fudge_lj, 0.5)
    assert np.isclose(td.fudge_qq, 0.83333)
    return tfs
