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

import torch
from torch import LongTensor, Tensor

from byteff.forcefield import ff_kernels
from byteff.utils.definitions import (CHG_FACTOR, PROPERTORSION_TERMS, ParamTerm, TopoData, TopoParams, TopoTerm,
                                      fudgeLJ, fudgeQQ)


class CFF:

    default_terms = {
        TopoTerm.Bond, TopoTerm.Angle, TopoTerm.ProperTorsion, TopoTerm.ImproperTorsion, TopoTerm.Nonbonded14,
        TopoTerm.NonbondedAll
    }

    @staticmethod
    def _autograd_hessian(coords: torch.Tensor, grad: torch.Tensor):
        assert coords.requires_grad
        natoms = coords.shape[0]
        nconfs = coords.shape[1]

        Qmats = torch.eye(natoms * 3, dtype=coords.dtype, device=coords.device)
        grad_output = Qmats.reshape(-1, natoms, 3).unsqueeze(-2)  # [3*natoms, natoms, nconfs, 3]
        QTH = torch.autograd.grad(grad, coords, grad_outputs=grad_output, create_graph=True,
                                  is_grads_batched=True)[0]  # [3*natoms, natoms, nconfs, 3]
        QTH = QTH.transpose(1, 2).reshape(-1, nconfs, 3 * natoms)  # [3*natoms, nconf, 3*natoms]
        QTH = torch.moveaxis(QTH, 1, -1)
        assert QTH.shape == (3 * natoms, 3 * natoms, nconfs), f'{QTH.shape}'
        return QTH  # [3*natoms, 3*natoms, nconfs]

    @staticmethod
    def _autograd_partial_hessian(paired_coords: torch.Tensor, grad: torch.Tensor):
        npairs, nconfs, width, na = paired_coords.shape
        assert na == 3
        assert grad.shape == paired_coords.shape
        assert paired_coords.requires_grad

        grad_outputs = torch.eye(width * 3, dtype=grad.dtype, device=grad.device).unsqueeze(0).unsqueeze(0).expand(
            (npairs, nconfs, -1, -1))  # [npairs, nconfs, width*3, width*3]
        grad_outputs = grad_outputs.movedim(-1, 0).reshape(-1, npairs, nconfs, width,
                                                           3)  # [width*3, npairs, nconfs, width, 3]

        hessian = torch.autograd.grad(grad,
                                      paired_coords,
                                      grad_outputs=grad_outputs,
                                      create_graph=True,
                                      is_grads_batched=True)[0]  # [width*3, npairs, nconfs, width, 3]
        hessian = hessian.movedim(0, -3).reshape(npairs, nconfs, width, 3, width,
                                                 3)  # [npairs, nconfs, width, 3, width, 3]
        hessian = hessian.movedim(-2, -3).reshape(npairs, nconfs, width * width, 3 * 3)
        return hessian  # [npairs, nconfs, width * width, 3 * 3]

    @classmethod
    def calc_bond(cls,
                  topo_params: TopoParams,
                  coords: torch.Tensor,
                  calc_force=True,
                  calc_partial_hessian=False) -> dict[Tensor]:
        """
            coords: [natoms, nconfs, 3]
        """
        field = TopoTerm.Bond
        width = TopoData.width[field]
        index = topo_params[field].long()  # [nbonds, 2]
        if index.shape[0] == 0:
            return {f'{field.name}_energy': 0., f'{field.name}_forces': 0.}

        k = topo_params[ParamTerm.Bond_k]  # [nbonds, 1]
        b0 = topo_params[ParamTerm.Bond_length]  # [nbonds, 1]
        counts = topo_params.get_count(field).clone()  # [batch_size]
        nconfs = coords.shape[1]

        cc = [coords[index[:, i]].unsqueeze(-2) for i in range(width)]
        cc = torch.concat(cc, dim=-2)  # [nbonds, nconfs, width, 3]
        ccs = [cc[:, :, i] for i in range(width)]
        r12, r12vec = ff_kernels.get_distance_vec_0(*ccs)  # [nbonds, nconfs], [nbonds, nconfs, 3]
        pair_energy = 0.5 * k * (r12 - b0)**2  # [nbonds, nconfs]
        energy = ff_kernels.reduce_batch_energy(pair_energy, counts)  # [batch_size, nconfs]

        forces = torch.zeros_like(coords)  # [natoms, nconfs, 3]
        if calc_force:
            pair_forces = (k * (1 - b0 / r12)).unsqueeze(-1) * r12vec  # [nbonds, nconfs, 3]
            atom1_idxs = index[:, 0].unsqueeze(-1).unsqueeze(-1).expand(-1, nconfs, 3)
            atom2_idxs = index[:, 1].unsqueeze(-1).unsqueeze(-1).expand(-1, nconfs, 3)
            # IMPORTANT: negative sign
            forces.scatter_add_(0, atom1_idxs, pair_forces)
            forces.scatter_add_(0, atom2_idxs, -pair_forces)
        ret = {f'{field.name}_energy': energy, f'{field.name}_forces': forces}

        if calc_partial_hessian:
            pair_forces = torch.concat([pair_forces.unsqueeze(-2), -pair_forces.unsqueeze(-2)],
                                       dim=-2)  # [nbonds, nconfs, width, 3]
            hessian = cls._autograd_partial_hessian(cc, -pair_forces)
            ret[f'{field.name}_hessian'] = hessian
        return ret

    @classmethod
    def calc_angle(cls,
                   topo_params: TopoParams,
                   coords: torch.Tensor,
                   calc_force=True,
                   calc_partial_hessian=False) -> dict[Tensor]:
        field = TopoTerm.Angle
        width = TopoData.width[field]
        index = topo_params[field].long()  # [nangles, 2]
        if index.shape[0] == 0:
            return {f'{field.name}_energy': 0., f'{field.name}_forces': 0.}

        k = topo_params[ParamTerm.Angle_k]  # [nangles, 1]
        theta0 = torch.deg2rad(topo_params[ParamTerm.Angle_theta])  # [nangles, 1]
        counts = topo_params.get_count(field).clone()  # [batch_size]
        nconfs = coords.shape[1]

        cc = [coords[index[:, i]].unsqueeze(-2) for i in range(width)]
        cc = torch.concat(cc, dim=-2)  # [nangles, nconfs, width, 3]
        ccs = [cc[:, :, i] for i in range(width)]
        theta, f1, f3 = ff_kernels.get_angle_vec_0(*ccs, with_vec=calc_force)
        pair_energy = 0.5 * k * (theta - theta0)**2  # [nangles, nconfs]
        energy = ff_kernels.reduce_batch_energy(pair_energy, counts)  # [batch_size, nconfs]

        forces = torch.zeros_like(coords)  # [natoms, nconfs, 3]
        if calc_force:
            # [nbatch, nconfs, nangles]
            atom1_idxs = index[:, 0].unsqueeze(-1).unsqueeze(-1).expand(-1, nconfs, 3)
            atom2_idxs = index[:, 1].unsqueeze(-1).unsqueeze(-1).expand(-1, nconfs, 3)
            atom3_idxs = index[:, 2].unsqueeze(-1).unsqueeze(-1).expand(-1, nconfs, 3)
            fc = -(k * (theta - theta0)).unsqueeze(-1)  # [nangles, nconfs, 1]
            force1 = fc * f1
            force3 = fc * f3
            force2 = -force1 - force3
            forces.scatter_add_(0, atom1_idxs, force1)
            forces.scatter_add_(0, atom2_idxs, force2)
            forces.scatter_add_(0, atom3_idxs, force3)
        ret = {f'{field.name}_energy': energy, f'{field.name}_forces': forces}

        if calc_partial_hessian:
            pair_forces = torch.concat(
                [force1.unsqueeze(-2), force2.unsqueeze(-2),
                 force3.unsqueeze(-2)], dim=-2)  # [nangles, nconfs, width, 3]
            hessian = cls._autograd_partial_hessian(cc, -pair_forces)
            ret[f'{field.name}_hessian'] = hessian  # [nangles, nconfs, width * width, 3 * 3]
        return ret

    @classmethod
    def _calc_dihedral_energy_forces(cls,
                                     coords,
                                     atomidx,
                                     counts,
                                     k,
                                     periodicity,
                                     phase,
                                     calc_force=True,
                                     calc_partial_hessian=False) -> tuple[Tensor, Tensor]:
        nconfs = coords.shape[1]
        nterms = k.shape[-1]
        k = k.unsqueeze(1).expand(-1, nconfs, -1)  # [ndihedrals, nconfs, nterms]
        periodicity = periodicity.unsqueeze(1).expand(-1, nconfs, -1)  # [ndihedrals, nconfs, nterms]
        phase = phase.unsqueeze(1).expand(-1, nconfs, -1)  # [ndihedrals, nconfs, nterms]

        width = 4
        cc = [coords[atomidx[:, i]].unsqueeze(-2) for i in range(width)]
        cc = torch.concat(cc, dim=-2)  # [ndihedrals, nconfs, width, 3]
        ccs = [cc[:, :, i] for i in range(width)]
        theta, f1, f2, f3, f4 = ff_kernels.get_dihedral_angle_vec_0(*ccs, with_vec=calc_force)
        theta_expanded = theta.unsqueeze(-1).expand(-1, -1, nterms)  # [ndihedrals, nconfs, nterms]
        dtheta = periodicity * theta_expanded - phase  # [ndihedrals, nconfs, nterms]
        # [ndihedrals, nconfs]
        dihedral_energy = torch.sum(k * (1 + torch.cos(dtheta)), dim=-1)
        energy = ff_kernels.reduce_batch_energy(dihedral_energy, counts)  # [batch_size, nconfs]

        forces = torch.zeros_like(coords)  # [natoms, nconfs, 3]
        if calc_force:
            # [ndihedrals, nconfs, 1]
            force_prefac = torch.sum(k * periodicity * torch.sin(dtheta), dim=-1).unsqueeze(-1)
            force1 = force_prefac * f1  # [ndihedrals, nconfs, 3]
            force2 = force_prefac * f2  # [ndihedrals, nconfs, 3]
            force3 = force_prefac * f3  # [ndihedrals, nconfs, 3]
            force4 = force_prefac * f4  # [ndihedrals, nconfs, 3]

            # [ndihedrals, nconfs, 3]
            atom1_idxs = atomidx[:, 0].unsqueeze(-1).unsqueeze(-1).expand(-1, nconfs, 3)
            atom2_idxs = atomidx[:, 1].unsqueeze(-1).unsqueeze(-1).expand(-1, nconfs, 3)
            atom3_idxs = atomidx[:, 2].unsqueeze(-1).unsqueeze(-1).expand(-1, nconfs, 3)
            atom4_idxs = atomidx[:, 3].unsqueeze(-1).unsqueeze(-1).expand(-1, nconfs, 3)

            # [ndihedrals, nconfs, 3] -> [natoms, nconfs, 3]
            forces.scatter_add_(0, atom1_idxs, force1)
            forces.scatter_add_(0, atom2_idxs, force2)
            forces.scatter_add_(0, atom3_idxs, force3)
            forces.scatter_add_(0, atom4_idxs, force4)
        if calc_partial_hessian:
            pair_forces = torch.concat(
                [force1.unsqueeze(-2),
                 force2.unsqueeze(-2),
                 force3.unsqueeze(-2),
                 force4.unsqueeze(-2)], dim=-2)  # [ndihedrals, nconfs, width, 3]
            # print(pair_forces)
            hessian = cls._autograd_partial_hessian(cc, -pair_forces)
        else:
            hessian = None

        return energy, forces, hessian

    @classmethod
    def calc_propertorsion(cls,
                           topo_params: TopoParams,
                           coords: torch.Tensor,
                           calc_force=True,
                           calc_partial_hessian=False):
        field = TopoTerm.ProperTorsion
        index = topo_params[field].long()  # [npropers, 4]
        if index.shape[0] == 0:
            return {f'{field.name}_energy': 0., f'{field.name}_forces': 0.}

        proper_n = [n + 1 for n in range(PROPERTORSION_TERMS)]
        k = topo_params[ParamTerm.ProperTorsion_k]  # [npropers, len(proper_n)]
        periodicity = torch.tensor(proper_n, dtype=coords.dtype,
                                   device=coords.device).unsqueeze(0).expand(k.shape[0], -1)
        phase = (((torch.tensor(proper_n, device=coords.device, dtype=torch.int64) + 1) % 2) * torch.pi).to(
            coords.dtype).unsqueeze(0).expand(k.shape[0], -1)
        counts = topo_params.get_count(field).clone()  # [batch_size]

        energy, forces, hessian = cls._calc_dihedral_energy_forces(coords, index, counts, k, periodicity, phase,
                                                                   calc_force, calc_partial_hessian)
        ret = {f'{field.name}_energy': energy, f'{field.name}_forces': forces}
        if calc_partial_hessian:
            ret[f'{field.name}_hessian'] = hessian
        return ret

    @staticmethod
    def calc_proper_theta(coords: torch.Tensor, index: torch.Tensor):
        theta, _, _, _, _ = ff_kernels.get_dihedral_angle_vec(coords, index, with_vec=False)
        return theta

    @classmethod
    def calc_impropertorsion(cls,
                             topo_params: TopoParams,
                             coords: torch.Tensor,
                             calc_force=True,
                             calc_partial_hessian=False):
        field = TopoTerm.ImproperTorsion
        index = topo_params[field].long()  # [nimpropers, 4]
        if index.shape[0] == 0:
            return {f'{field.name}_energy': 0., f'{field.name}_forces': 0.}

        k = topo_params[ParamTerm.ImproperTorsion_k]  # [nimpropers, 1]
        periodicity = torch.ones_like(k) * 2.
        phase = torch.ones_like(k) * torch.pi
        counts = topo_params.get_count(field).clone()  # [batch_size]

        # add all three improper dihedrals to improper list
        index1 = index[:, torch.LongTensor([0, 2, 3, 1])]
        index2 = index[:, torch.LongTensor([0, 3, 1, 2])]
        index = torch.concat([index, index1, index2], dim=0)  # [3 * nimpropers, 4]
        k = k.repeat(3, 1) / 3.0
        periodicity = periodicity.repeat(3, 1)
        phase = phase.repeat(3, 1)
        nmol = counts.shape[0]
        counts = counts.repeat(3)

        energy, forces, hessian = cls._calc_dihedral_energy_forces(coords,
                                                                   index,
                                                                   counts,
                                                                   k,
                                                                   periodicity,
                                                                   phase,
                                                                   calc_force=calc_force,
                                                                   calc_partial_hessian=calc_partial_hessian)
        energy = sum(torch.split(energy, nmol))
        ret = {f'{field.name}_energy': energy, f'{field.name}_forces': forces}
        if calc_partial_hessian:
            nimpropers, nconfs = hessian.shape[0] // 3, hessian.shape[1]
            h0, h1, h2 = torch.split(hessian, nimpropers, 0)  # [nimpropers, nconfs, width * width, 3 * 3]
            h1 = h1.reshape(nimpropers, nconfs, 4, 4, 9)[:, :,
                                                         torch.LongTensor([0, 3, 1, 2])][:, :, :,
                                                                                         torch.LongTensor([0, 3, 1, 2])]
            h2 = h2.reshape(nimpropers, nconfs, 4, 4, 9)[:, :,
                                                         torch.LongTensor([0, 2, 3, 1])][:, :, :,
                                                                                         torch.LongTensor([0, 2, 3, 1])]
            hessian = h0 + h1.reshape(nimpropers, nconfs, 4 * 4, 9) + h2.reshape(nimpropers, nconfs, 4 * 4, 9)
            ret[f'{field.name}_hessian'] = hessian
        return ret

    @staticmethod
    def _calc_nonbonded_energy_forces(coords: Tensor,
                                      atomidx: LongTensor,
                                      counts,
                                      sigma,
                                      epsilon,
                                      partial_charges,
                                      calc_force=True) -> tuple[Tensor, Tensor]:
        nconfs = coords.shape[1]
        a1_idxs, a2_idxs = atomidx[:, 0], atomidx[:, 1]  # [npairs]
        charge1, charge2 = partial_charges[a1_idxs], partial_charges[a2_idxs]
        sigma1, sigma2 = sigma[a1_idxs], sigma[a2_idxs]
        epsilon1, epsilon2 = epsilon[a1_idxs], epsilon[a2_idxs]

        # combination rule:
        # The combining_rules attribute (default: "none") currently only supports "Lorentz-Berthelot",
        # which specifies the geometric mean of epsilon and arithmetic mean of sigma.
        epsilon = torch.sqrt(epsilon1 * epsilon2)  # [npairs]
        sigma = 0.5 * (sigma1 + sigma2)  # [npairs]

        # [npairs, nconfs], [npairs, nconfs, 3]
        r12, r12vec = ff_kernels.get_distance_vec(coords, atomidx)
        invr12 = 1.0 / r12  # [npairs, nconfs]

        sigma_invr12 = sigma.unsqueeze(-1) * invr12  # [npairs, nconfs]
        # U(r) = 4*epsilon*((sigma/r)^12 - (sigma/r)^6)
        u6 = torch.pow(sigma_invr12, 6)  # [npairs, nconfs]
        u12 = torch.square(u6)  # [npairs, nconfs]

        chg_energy = CHG_FACTOR * invr12 * (charge1 * charge2).unsqueeze(-1)  # [npairs, nconfs]
        vdW_energy = 4 * epsilon.unsqueeze(-1) * (u12 - u6)  # [npairs, nconfs]
        chg_energy = ff_kernels.reduce_batch_energy(chg_energy, counts)  # [batch_size, nconfs]
        vdW_energy = ff_kernels.reduce_batch_energy(vdW_energy, counts)  # [batch_size, nconfs]

        chg_forces = torch.zeros_like(coords)
        vdW_forces = torch.zeros_like(coords)
        if calc_force:
            # IMPORTANT: negative sign
            # [npairs, nconfs, 3]
            pair_force = -CHG_FACTOR * (torch.pow(invr12, 3) * (charge1 * charge2).unsqueeze(-1)).unsqueeze(-1) * r12vec
            chg_forces.scatter_add_(0, a1_idxs.unsqueeze(-1).unsqueeze(-1).expand(-1, nconfs, 3), pair_force)
            chg_forces.scatter_add_(0, a2_idxs.unsqueeze(-1).unsqueeze(-1).expand(-1, nconfs, 3), -pair_force)

            # IMPORTANT: negative sign
            pair_force = 4 * ((epsilon.unsqueeze(-1) * (-12 * u12 + 6 * u6)) * (invr12 * invr12)).unsqueeze(-1) * r12vec
            vdW_forces.scatter_add_(0, a1_idxs.unsqueeze(-1).unsqueeze(-1).expand(-1, nconfs, 3), pair_force)
            vdW_forces.scatter_add_(0, a2_idxs.unsqueeze(-1).unsqueeze(-1).expand(-1, nconfs, 3), -pair_force)
        return chg_energy, chg_forces, vdW_energy, vdW_forces

    @classmethod
    def calc_nonbonded14(cls,
                         topo_params: TopoParams,
                         coords: torch.Tensor,
                         calc_force=True,
                         calc_partial_hessian=False):
        field = TopoTerm.Nonbonded14
        index = topo_params[field].long()  # [npairs, 2]
        counts = topo_params.get_count(field).clone()  # [batch_size]
        sigma = topo_params[ParamTerm.Sigma].squeeze(-1)  # [natoms]
        epsilon = topo_params[ParamTerm.Epsilon].squeeze(-1)  # [natoms]
        partial_charges = topo_params[ParamTerm.Charge].squeeze(-1)  # [natoms]

        chg, chg_f, vdW, vdW_f = cls._calc_nonbonded_energy_forces(coords, index, counts, sigma, epsilon,
                                                                   partial_charges, calc_force)

        # nonbonded14 energy and forces are scaled by fudgeQQ and fudgeLJ
        results = {
            f'{field.name}_energy': chg * fudgeQQ + vdW * fudgeLJ,
            f'{field.name}_forces': chg_f * fudgeQQ + vdW_f * fudgeLJ,
        }
        return results

    @classmethod
    def calc_nonbondedall(cls,
                          topo_params: TopoParams,
                          coords: torch.Tensor,
                          calc_force=True,
                          calc_partial_hessian=False):
        field = TopoTerm.NonbondedAll
        index = topo_params[field].long()  # [npairs, 2]
        counts = topo_params.get_count(field).clone()  # [batch_size]
        sigma = topo_params[ParamTerm.Sigma].squeeze(-1)  # [natoms]
        epsilon = topo_params[ParamTerm.Epsilon].squeeze(-1)  # [natoms]
        partial_charges = topo_params[ParamTerm.Charge].squeeze(-1)  # [natoms]

        chg, chg_f, vdW, vdW_f = cls._calc_nonbonded_energy_forces(coords, index, counts, sigma, epsilon,
                                                                   partial_charges, calc_force)
        results = {
            f'{field.name}_energy': chg + vdW,
            f'{field.name}_forces': chg_f + vdW_f,
        }
        return results

    @classmethod
    def energy_force(cls,
                     topo_params: TopoParams,
                     coords: torch.Tensor,
                     calc_terms=None,
                     calc_force=True,
                     calc_hessian=False,
                     calc_partial_hessian=False,
                     confmask: torch.Tensor = None) -> dict[str, Tensor]:
        if calc_terms is None:
            calc_terms = cls.default_terms
        else:
            calc_terms = set(calc_terms)
            assert calc_terms.issubset(cls.default_terms)

        if coords.shape[1] > 1:
            assert confmask is not None, "must provide confmask for coords containing more than one conformer!"

        natoms = topo_params.get_count(TopoTerm.Atom)
        results = {
            'energy': torch.zeros([len(natoms), coords.shape[1]], dtype=coords.dtype, device=coords.device),
            'forces': torch.zeros_like(coords)
        }

        if calc_hessian or calc_partial_hessian:
            coords.requires_grad = True
            calc_force = True

        for term in calc_terms:
            ret = getattr(cls, f"calc_{term.name.lower()}")(topo_params, coords, calc_force, calc_partial_hessian)
            results.update(ret)
            results['energy'] += ret[f'{term.name}_energy']
            results['forces'] += ret[f'{term.name}_forces']

        if calc_hessian:
            hessian = cls._autograd_hessian(coords, -results['forces'])
            results['hessian'] = hessian

        if confmask is not None:
            batch = torch.repeat_interleave(torch.arange(len(natoms), device=natoms.device), natoms)
            confmask_forces = ff_kernels._batch_to_atoms(confmask, batch.unsqueeze(-1).expand(-1, coords.shape[1]))
            for k in results:
                if k.endswith('energy'):
                    results[k] *= confmask
                else:
                    results[k] *= confmask_forces
        return results
