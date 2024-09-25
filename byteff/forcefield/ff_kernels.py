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


def _batch_to_atoms(src: Tensor, batch: LongTensor) -> Tensor:
    """[bs, nconfs] -> [natoms, nconfs, 3]"""
    ret = torch.gather(src, 0, batch)
    ret = ret.unsqueeze(-1).expand(-1, -1, 3)
    return ret


def _batch_reduce(src: Tensor, batch: LongTensor, reduce: str = 'sum'):
    """[natoms, nconfs] -> [bs, nconfs]"""
    ret = torch.zeros((torch.max(batch) + 1, batch.shape[1]), dtype=src.dtype, device=src.device)
    ret.scatter_reduce_(0, batch, src, reduce)  # [bs, nconfs]
    return ret


def _batch_dot(a: Tensor, b: Tensor, batch: LongTensor, keep_dim=False) -> Tensor:
    """reduce according to batch tensor
    a, b: [natoms, nconfs, 3]
    batch: [natoms, nconfs]
    """
    prod = torch.sum(a * b, dim=-1)  # [natoms, nconfs]
    prod_sum = _batch_reduce(prod, batch)  # [bs, nconfs]
    if keep_dim:
        return _batch_to_atoms(prod_sum, batch)  # [natoms, nconfs, 3]
    else:
        return prod_sum


def reduce_batch_energy(pair_energy: Tensor, counts: LongTensor) -> Tensor:
    """Reduce pairwise energy by batch
    args:
        pair_energy: [npairs, nconfs]
        counts: [batch_size]
    return:
        energy: [batch_size, nconfs]
    """
    nconfs = pair_energy.shape[1]
    energy = torch.zeros((len(counts), nconfs), dtype=pair_energy.dtype, device=pair_energy.device)
    reduce_idx = torch.repeat_interleave(torch.arange(len(counts), device=counts.device),
                                         counts).unsqueeze(-1).expand(-1, nconfs)
    energy.scatter_add_(0, reduce_idx, pair_energy)  # [batch_size, nconfs]
    return energy


def get_distance_vec_0(r1, r2) -> tuple[Tensor, Tensor]:
    r12vec = r2 - r1
    r12 = torch.linalg.vector_norm(r12vec, dim=-1)
    return r12, r12vec


def get_distance_vec(coords: Tensor, atomidx: LongTensor) -> tuple[Tensor, Tensor]:
    """Compute distance between atoms
    args:
        coords: [natoms, nconfs, 3]
        atomidx: [npairs, 2]
    return:
        r12: [npairs, nconfs]
        r12vec: [npairs, nconfs, 3]
    """
    r1, r2 = coords[atomidx[:, 0]], coords[atomidx[:, 1]]
    return get_distance_vec_0(r1, r2)


def get_angle_vec_0(r0, r1, r2, with_vec: bool = True):
    v0 = r1 - r0  # [nangles, nconfs, 3]
    v1 = r1 - r2  # [nangles, nconfs, 3]
    cross = torch.linalg.cross(v0, v1, dim=-1)  # [nangles, nconfs, 3]
    dot_v0_v1 = torch.linalg.vecdot(v0, v1)
    # atan2 generates nan when computing hessian for exact linear angle.
    angle = torch.atan2(torch.linalg.vector_norm(cross, dim=-1), dot_v0_v1)

    if with_vec:
        # rp becomes singular when cross=0 (v0//v1), then f1/f3 become nan
        r0 = torch.linalg.vecdot(v0, v0)
        r1 = torch.linalg.vecdot(v1, v1)
        rp = torch.linalg.vector_norm(cross, dim=-1)  # [nangle, nconfs, 1]
        f1 = -torch.linalg.cross(v0, cross, dim=-1) / (r0 * rp).unsqueeze(-1)
        f3 = -torch.linalg.cross(cross, v1, dim=-1) / (r1 * rp).unsqueeze(-1)
    else:
        f1 = f3 = torch.zeros_like(r0)

    return angle, f1, f3


def get_angle_vec(coords: Tensor, atomidx: LongTensor, with_vec: bool = True):
    '''Calculate angle
    reference: 
        https://github.com/openmm/openmm/blob/81271f93abeca82af709e1b288b2fc4aa0c272d5/platforms/common/src/kernels/angleForce.cc
    known issue:
        for exactly linear angle (either 0 or pi):
        the direction of force is physically undefined, so grad(force, coords) is nan
        the magnitude of force is still defined
        reference:
        1. Swope, W. C. & Ferguson, D. M. Alternative expressions for energies and forces due to angle bending and torsional energy. Journal of Computational Chemistry 13, 585-594 (1992).

    args:
        coords: [natoms, nconfs, 3]
        atomidx: [nangles, 3]
    return:
        angle: [nangles, nconfs]
        f1, f3: [nangle, nconfs, 3]
    '''
    r0, r1, r2 = coords[atomidx[:, 0]], coords[atomidx[:, 1]], coords[atomidx[:, 2]]
    return get_angle_vec_0(r0, r1, r2, with_vec=with_vec)


def get_dihedral_angle_vec_0(r0, r1, r2, r3, with_vec: bool = True):

    # use the method in gromacs 1234 <-> ijkl
    r_ij = r1 - r0
    r_kj = r1 - r2
    r_kl = r3 - r2
    m = torch.linalg.cross(r_ij, r_kj, dim=-1)
    n = torch.linalg.cross(r_kj, r_kl, dim=-1)
    w = torch.linalg.cross(m, n, dim=-1)
    wlen = torch.linalg.vector_norm(w, dim=-1)  # [ndihedrals, nconfs]
    s = torch.linalg.vecdot(m, n)
    phi = torch.atan2(wlen, s)  # [ndihedrals, nconfs]
    ipr = torch.linalg.vecdot(r_ij, n)  # [ndihedrals, nconfs]
    ipr = torch.where(torch.abs(ipr) > torch.finfo().eps, ipr, 1.0)
    phi = -phi * torch.sign(ipr)  # right hand sign

    if with_vec:
        iprm = torch.linalg.vecdot(m, m)
        iprn = torch.linalg.vecdot(n, n)
        nrkj2 = torch.linalg.vecdot(r_kj, r_kj)

        nrkj_1 = torch.rsqrt(nrkj2)  # [ndihedrals, nconfs]
        nrkj_2 = torch.square(nrkj_1)  # [ndihedrals, nconfs]
        nrkj = nrkj2 * nrkj_1  # [ndihedrals, nconfs]
        a = -nrkj / iprm  # [ndihedrals, nconfs]
        f_i = -a.unsqueeze(-1) * m  # [ndihedrals, nconfs, 3]
        b = nrkj / iprn  # [ndihedrals, nconfs]
        f_l = -b.unsqueeze(-1) * n  # [ndihedrals, nconfs, 3]
        p = torch.linalg.vecdot(r_ij, r_kj)  # [ndihedrals, nconfs]
        p *= nrkj_2  # [ndihedrals, nconfs]
        q = torch.linalg.vecdot(r_kl, r_kj)  # [ndihedrals, nconfs]
        q *= nrkj_2  # [ndihedrals, nconfs]

        uvec = p.unsqueeze(-1) * f_i  # [ndihedrals, nconfs, 3]
        vvec = q.unsqueeze(-1) * f_l  # [ndihedrals, nconfs, 3]
        svec = uvec - vvec  # [ndihedrals, nconfs, 3]
        f_j = (f_i - svec)  # [ndihedrals, nconfs, 3]
        f_k = (f_l + svec)  # [ndihedrals, nconfs, 3]
    else:
        f_i = f_j = f_k = f_l = torch.zeros_like(r0)

    return phi, f_i, -f_j, -f_k, f_l


def get_dihedral_angle_vec(coords: Tensor, atomidx: LongTensor, with_vec: bool = True):
    ''' calculate dihedral angle
        coords: [..., nconfs, natoms, 3]
        atomidx: [..., nangles, 4]
        reference: 
        https://gitlab.com/gromacs/gromacs/-/blob/8cadd7d248c88c2efc9f3e01b4ffd139cf395376/src/gromacs/listed_forces/listed_forces_gpu_internal.cu
    '''
    # [ndihedrals, nconfs, 3] * 4
    r0, r1, r2, r3 = coords[atomidx[..., 0]], coords[atomidx[..., 1]], coords[atomidx[..., 2]], coords[atomidx[..., 3]]

    return get_dihedral_angle_vec_0(r0, r1, r2, r3, with_vec=with_vec)


def dihedral_jacobian(coords: torch.Tensor, torsion_ids: torch.LongTensor):
    torsion_ids = torsion_ids.long()
    phi, f_i, f_j, f_k, f_l = get_dihedral_angle_vec(coords, torsion_ids.T)
    jac = torch.zeros_like(coords)  # [natoms, nconfs, 3]
    jac.scatter_add_(0, torsion_ids[0].unsqueeze(-1).unsqueeze(-1).expand(-1, coords.shape[1], 3), f_i)
    jac.scatter_add_(0, torsion_ids[1].unsqueeze(-1).unsqueeze(-1).expand(-1, coords.shape[1], 3), f_j)
    jac.scatter_add_(0, torsion_ids[2].unsqueeze(-1).unsqueeze(-1).expand(-1, coords.shape[1], 3), f_k)
    jac.scatter_add_(0, torsion_ids[3].unsqueeze(-1).unsqueeze(-1).expand(-1, coords.shape[1], 3), f_l)
    return phi, jac
