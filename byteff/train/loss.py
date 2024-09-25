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
from enum import Enum
from functools import partial

import torch
from torch import Tensor
from torch.nn.functional import cross_entropy

from byteff.data import MolData
from byteff.forcefield import CFF
from byteff.forcefield.ff_kernels import (_batch_reduce, dihedral_jacobian, get_angle_vec, get_distance_vec)
from byteff.forcefield.ffopt import ConstraintFFopt
from byteff.utils import to_global_idx
from byteff.utils.definitions import (ParamData, ParamTerm, TopoData, TopoParams, TopoTerm)

logger = logging.getLogger(__name__)


def set_grad_max(grad: Tensor, max_step: float):
    if grad.abs().max() > max_step:
        grad *= max_step / grad.abs().max()
    return grad


def set_params_max(preds: TopoParams, max_config: dict[ParamTerm, float]):
    # set max grad for specified params
    # a param will be detached if not in max_config
    # 0. means zero grad, < 0. means no limit

    new_tp = preds.copy()
    for term in ParamTerm:
        v = new_tp[term]
        if term in max_config:
            v = v.clone()
            if max_config[term] >= 0. and v.requires_grad:
                v.register_hook(partial(set_grad_max, max_step=max_config[term]))
                new_tp[term] = v
        else:
            new_tp[term] = v.detach()
    return new_tp


def set_trainable_terms(topo_params: TopoParams, trainable_terms: list[ParamTerm], max_config: dict[str, float] = None):
    _max_config = {}
    for term in trainable_terms:
        if max_config is not None and term.name in max_config:
            _max_config[term] = max_config[term.name]
        else:
            _max_config[term] = torch.finfo().max
    topo_params = set_params_max(topo_params, _max_config)
    return topo_params


def calc_conf_mean(src, confmask):
    x = src * confmask  # [nmols * nconfs]
    return torch.sum(x, dim=-1) / torch.sum(confmask, dim=-1)


def soft_mse(diff: Tensor, max_val: float, keep_dim=False):
    l = diff**2
    scale = torch.tanh(l / max_val) * max_val / torch.where(abs(l) < torch.finfo().eps, torch.finfo().eps, l)
    scale = scale.detach()
    if keep_dim:
        return l * scale
    else:
        return torch.mean(l * scale)


def calc_circstd(graph):
    propers = CFF.calc_proper_theta(graph.coords, graph.ProperTorsion_index.T.long())
    confmask = graph.confmask
    confmask = torch.repeat_interleave(confmask, graph.nProperTorsion, 0)

    sin_prop = torch.sin(propers) * confmask
    cos_prop = torch.cos(propers) * confmask
    sin_prop = torch.sum(sin_prop, dim=-1) / torch.sum(confmask, dim=-1)
    cos_prop = torch.sum(cos_prop, dim=-1) / torch.sum(confmask, dim=-1)

    # clamp to avoid nan caused by float precision
    R2 = torch.clamp(torch.square(sin_prop) + torch.square(cos_prop),
                     max=1.0 - torch.finfo().eps * 10.,
                     min=torch.finfo().eps * 10.)
    std = torch.sqrt(-torch.log(R2))

    return std


def mask_by_circstd(preds: dict[ParamTerm, Tensor], graph: MolData, theshold: float):

    prop_std = calc_circstd(graph)
    p = preds[ParamTerm.ProperTorsion_k]
    p1 = p.clone().detach()
    mask = torch.where(prop_std > theshold, 1., 0.).unsqueeze(-1)
    preds[ParamTerm.ProperTorsion_k] = p * mask + p1 * (1 - mask)
    return preds


class TorsionLossType(Enum):
    mse = 1
    boltzmann_mse = 2
    boltzmann_soft_mse = 3
    l1_norm = 4
    l1_rotate = 5
    l2_norm = 6
    ortho_force_mse = 7
    ortho_force_boltzmann_mse = 8


def torsion_loss(topo_params: dict[ParamTerm, Tensor],
                 graph: MolData,
                 loss_type: TorsionLossType,
                 max_config: dict[str, float] = None,
                 **kwargs):
    topo_params = set_trainable_terms(topo_params, [ParamTerm.ProperTorsion_k], max_config)

    if 'mask_by_circstd_threshold' in kwargs:
        mask_by_circstd(topo_params, graph, kwargs['mask_by_circstd_threshold'])

    if loss_type == TorsionLossType.l1_norm:
        loss = torch.mean(torch.abs(topo_params[ParamTerm.ProperTorsion_k]))

    elif loss_type == TorsionLossType.l2_norm:
        loss = torch.mean(torch.square(topo_params[ParamTerm.ProperTorsion_k]))

    else:

        if kwargs.get('use_label_params', False):
            topo_params[ParamTerm.Bond_k] = graph.Bond_k_label
            topo_params[ParamTerm.Bond_length] = graph.Bond_length_label
            topo_params[ParamTerm.Angle_k] = graph.Angle_k_label
            topo_params[ParamTerm.Angle_theta] = graph.Angle_theta_label
            topo_params[ParamTerm.ImproperTorsion_k] = graph.ImproperTorsion_k_label
            # preds[ParamTerm.ProperTorsion_k] = graph.ProperTorsion_k_label

        confmask = graph.confmask  # [nbatch, nconfs]
        label_energy = graph.label_energy  # [nbatch, nconfs]
        results = CFF.energy_force(topo_params,
                                   graph.coords,
                                   confmask=graph.confmask,
                                   calc_force=loss_type
                                   in {TorsionLossType.ortho_force_mse, TorsionLossType.ortho_force_boltzmann_mse})
        pred_energy = results['energy']

        if loss_type == TorsionLossType.mse:
            shift = calc_conf_mean(label_energy - pred_energy, confmask).unsqueeze(-1).detach()  # [nbatch, 1]
            loss = torch.square(pred_energy - label_energy + shift)  # [nbatch, nconfs]
            loss = torch.mean(calc_conf_mean(loss, confmask))

        elif loss_type == TorsionLossType.boltzmann_mse:
            clamp = kwargs.pop('clamp', 2)  # unit kcal/mol
            decay = kwargs.pop('decay', 2)  # unit kcal/mol

            label_min = torch.min(torch.where(confmask > 0.5, label_energy,
                                              torch.finfo().max * torch.ones_like(label_energy)),
                                  dim=-1).values
            pred_min = torch.min(torch.where(confmask > 0.5, pred_energy,
                                             torch.finfo().max * torch.ones_like(pred_energy)),
                                 dim=-1).values
            # label_min = calc_conf_mean(label_energy)
            # pred_min = calc_conf_mean(pred_energy)

            label_energy_aligned = label_energy - label_min.unsqueeze(-1)
            pred_energy_aligned = pred_energy - pred_min.unsqueeze(-1)
            scale = torch.exp(
                torch.clamp((clamp - torch.minimum(label_energy_aligned, pred_energy_aligned)) / decay,
                            max=0)).detach()
            shift = (calc_conf_mean(
                (label_energy - pred_energy) * scale, confmask) / calc_conf_mean(scale, confmask)).detach()
            loss = torch.square(pred_energy - label_energy + shift.unsqueeze(-1)) * scale
            loss = calc_conf_mean(loss, confmask)

            # nb_e = results['Nonbonded_energy']
            # nb_mean = calc_conf_mean(nb_e)
            # nb_std = calc_conf_mean((nb_e - nb_mean.unsqueeze(-1))**2)**0.5

            # ub = kwargs.pop('upper_bound', torch.finfo().max)
            # nbstd_ub = kwargs.pop('nb_std_bound', torch.finfo().max)
            # if ub or nbstd_ub:
            #     # loss = torch.where(torch.logical_and(nb_std > nbstd_ub, loss > ub), 0., loss)
            #     loss = torch.where(torch.logical_or(nb_std > nbstd_ub, loss > ub), 0., loss)

            #     counts = torch.sum(torch.where(loss > 0., 1., 0.))
            #     # print(counts)
            #     loss = torch.sum(loss) / counts
            # else:
            loss = torch.mean(loss)

        elif loss_type == TorsionLossType.boltzmann_soft_mse:
            clamp = kwargs.pop('clamp', 2)  # unit kcal/mol
            decay = kwargs.pop('decay', 2)  # unit kcal/mol
            max_val = kwargs.pop('max', 100.)
            label_min = torch.min(torch.where(confmask > 0.5, label_energy,
                                              torch.finfo().max * torch.ones_like(label_energy)),
                                  dim=-1).values
            pred_min = torch.min(torch.where(confmask > 0.5, pred_energy,
                                             torch.finfo().max * torch.ones_like(pred_energy)),
                                 dim=-1).values
            label_energy_aligned = label_energy - label_min.unsqueeze(-1)
            pred_energy_aligned = pred_energy - pred_min.unsqueeze(-1)
            scale = torch.exp(
                torch.clamp((clamp - torch.minimum(label_energy_aligned, pred_energy_aligned)) / decay,
                            max=0)).detach()
            shift = (calc_conf_mean(
                (label_energy - pred_energy) * scale, confmask) / calc_conf_mean(scale, confmask)).detach()
            loss = torch.square(pred_energy - label_energy + shift.unsqueeze(-1)) * scale
            loss = calc_conf_mean(loss, confmask)
            scale = torch.tanh(loss / max_val) * max_val / torch.where(
                abs(loss) < torch.finfo().eps,
                torch.finfo().eps, loss)
            scale = scale.detach()
            loss = torch.mean(loss * scale)

        elif loss_type == TorsionLossType.ortho_force_mse:
            coords = graph.coords.detach()
            batch = graph.batch.detach().unsqueeze(-1).expand(-1, coords.shape[1])
            torsion_index = graph.torsion_index
            _, jac = dihedral_jacobian(coords, torsion_index)
            force = ConstraintFFopt.adjust_grad(results['forces'], jac, batch)
            loss = torch.mean(torch.square(force))

        elif loss_type == TorsionLossType.ortho_force_boltzmann_mse:
            clamp = kwargs.pop('clamp', 2)  # unit kcal/mol
            decay = kwargs.pop('decay', 2)  # unit kcal/mol
            coords = graph.coords.detach()
            batch = graph.batch.detach().unsqueeze(-1).expand(-1, coords.shape[1])
            torsion_index = graph.torsion_index
            _, jac = dihedral_jacobian(coords, torsion_index)
            force = ConstraintFFopt.adjust_grad(results['forces'], jac, batch)
            loss = torch.mean(torch.square(force), dim=-1)  # [natoms, nconfs]
            # print(torch.max(loss).item(), torch.argmax(loss) // loss.shape[1], torch.argmax(loss) % loss.shape[1])

            # _l = ConstraintFFopt.adjust_grad(results['Bond_forces'], jac, batch)
            # ll = torch.mean(torch.square(_l), dim=-1)  # [natoms, nconfs]
            # print('Bond', torch.max(ll).item(), torch.argmax(ll) // ll.shape[1], torch.argmax(ll) % ll.shape[1])

            # _l = ConstraintFFopt.adjust_grad(results['Angle_forces'], jac, batch)
            # ll = torch.mean(torch.square(_l), dim=-1)  # [natoms, nconfs]
            # print('Angle', torch.max(ll).item(), torch.argmax(ll) // ll.shape[1], torch.argmax(ll) % ll.shape[1])

            # _l = ConstraintFFopt.adjust_grad(results['ProperTorsion_forces'], jac, batch)
            # ll = torch.mean(torch.square(_l), dim=-1)  # [natoms, nconfs]
            # print('ProperTorsion', torch.max(ll).item(), torch.argmax(ll) // ll.shape[1], torch.argmax(ll) % ll.shape[1])

            # _l = ConstraintFFopt.adjust_grad(results['Nonbonded_forces'], jac, batch)
            # ll = torch.mean(torch.square(_l), dim=-1)  # [natoms, nconfs]
            # print('Nonbonded', torch.max(ll).item(), torch.argmax(ll) // ll.shape[1], torch.argmax(ll) % ll.shape[1])

            loss = _batch_reduce(loss, batch, 'mean')  # [bs, nconfs]

            label_min = torch.min(torch.where(confmask > 0.5, label_energy,
                                              torch.finfo().max * torch.ones_like(label_energy)),
                                  dim=-1).values
            label_energy_aligned = label_energy - label_min.unsqueeze(-1)
            scale = torch.exp(torch.clamp((clamp - label_energy_aligned) / decay, max=0)).detach()

            loss = loss * scale * confmask

            loss = torch.mean(loss)

        else:
            raise NotImplementedError()

    return loss


class HessianLossType(Enum):
    Bond_k_MSPE = 1
    Angle_k_MSPE = 2
    ProperTorsion_k_MSE = 3
    ImproperTorsion_k_MSE = 4
    Sigma_MSE = 5
    Epsilon_MSE = 6
    Charge_MSE = 7
    Bond_Energy = 8
    Angle_Energy = 9
    ImproperTorsion_Energy = 12
    Force_MSE = 10
    Force_Soft_MSE = 11
    Partial_Hessian_MAPE = 13


def hessian_loss(topo_params: TopoParams,
                 graph: MolData,
                 loss_type: HessianLossType,
                 max_config: dict[str, float] = None,
                 **kwargs):

    if '_Energy' in loss_type.name:
        term = getattr(TopoTerm, loss_type.name[:-len('_Energy')])
        param_map = {
            TopoTerm.Bond: ParamTerm.Bond_length,
            TopoTerm.Angle: ParamTerm.Angle_theta,
            TopoTerm.ImproperTorsion: ParamTerm.ImproperTorsion_k
        }
        topo_params = set_trainable_terms(topo_params, [param_map[term]], max_config)

        preds = CFF.energy_force(topo_params, graph.coords, calc_terms=[term], calc_force=False)
        loss = preds[f'{term.name}_energy']
        loss = loss.sum() / getattr(graph, f'n{term.name}').sum()

    elif loss_type is HessianLossType.Partial_Hessian_MAPE:

        mask_threshold = kwargs.pop("mask_threshold", 1e4)
        topo_params = set_trainable_terms(topo_params,
                                          [ParamTerm.Bond_k, ParamTerm.Angle_k, ParamTerm.ImproperTorsion_k],
                                          max_config)

        terms = {TopoTerm.Bond, TopoTerm.Angle, TopoTerm.ProperTorsion, TopoTerm.ImproperTorsion}
        preds = CFF.energy_force(topo_params, graph.coords, calc_terms=terms, calc_partial_hessian=True)
        partial_hessian_label: torch.Tensor = graph.PartialHessian  # [nPartialHessian, nconfs, 9]
        partial_hessian_pred = torch.zeros_like(partial_hessian_label)
        shifts = getattr(graph, 'nPartialHessian')
        nconfs = partial_hessian_label.shape[1]
        for term in terms:
            width = TopoData.width[term]
            for i in range(width):
                for j in range(width):
                    if i == j:
                        continue
                    if abs(i - j) >= 3 and term is TopoTerm.ProperTorsion:
                        continue
                    rec_ij = getattr(graph, f'{term.name}_rec_{i}_{j}').long()  # [nterms]
                    counts = getattr(graph, f'n{term.name}')
                    idx = to_global_idx(rec_ij, shifts, counts).unsqueeze(-1).unsqueeze(-1).expand(-1, nconfs, 9)

                    ph = preds[f'{term.name}_hessian']
                    partial_hessian_pred.scatter_add_(0, idx, ph[:, :, i * width + j])

        bad_ids = torch.where(partial_hessian_pred.abs() > mask_threshold)[0]
        mask = torch.ones_like(partial_hessian_pred)
        if len(bad_ids) > 0:
            cshift = torch.cumsum(shifts, dim=0)
            bads = set()
            for idx in set(bad_ids.tolist()):
                bads.add(torch.where(idx < cshift)[0][0].item())
                mask[idx] = 0.
            for idx in bads:
                logger.warning(f'found bad partial hessian: {graph.name[idx]}')

        loss = (partial_hessian_pred - partial_hessian_label).abs() * mask  # [nPartialHessian, nconfs, 9]
        diag = partial_hessian_label[..., 0] + partial_hessian_label[..., 4] + partial_hessian_label[..., 8]
        loss = loss / torch.clamp(diag.abs().unsqueeze(-1), min=10.0)

        # bad_id = torch.argmax(loss) // 9
        # print(partial_hessian_label[bad_id])
        # print(partial_hessian_pred[bad_id])
        # print(loss[bad_id])
        # cshift = torch.cumsum(shifts, dim=0)
        # bad_id = torch.where(bad_id < cshift)[0][0].item()
        # print(graph.name[bad_id])

        loss = loss.mean()

    elif loss_type is HessianLossType.Force_MSE:
        topo_params = set_trainable_terms(topo_params, [ParamTerm.Bond_length, ParamTerm.Angle_theta], max_config)

        skip_nb_threshold = kwargs.pop("skip_large_nb", None)

        results = CFF.energy_force(topo_params, graph.coords, calc_force=True)
        forces = results['forces']
        if skip_nb_threshold is not None:
            nb_forces = results[f'{TopoTerm.Nonbonded14.name}_forces'] + results[f'{TopoTerm.NonbondedAll.name}_forces']
            batch = graph.batch[torch.where(nb_forces.abs() > skip_nb_threshold)[0]]
            for b in set(batch.tolist()):
                mask = torch.where(graph.batch == b, 0., 1.)
                forces *= mask.unsqueeze(-1).unsqueeze(-1)
        loss = torch.mean(torch.square(forces))

    elif loss_type is HessianLossType.Force_Soft_MSE:
        topo_params = set_trainable_terms(topo_params, [ParamTerm.Bond_length, ParamTerm.Angle_theta], max_config)

        max_val = kwargs.pop('max', 10.0)
        skip_nb_threshold = kwargs.pop("skip_large_nb", None)

        results = CFF.energy_force(topo_params, graph.coords, calc_force=True)
        forces = results['forces']
        # print(results['vdW_forces'].abs().max())
        if skip_nb_threshold is not None:
            nb_forces = results[f'{TopoTerm.Nonbonded14.name}_forces'] + results[f'{TopoTerm.NonbondedAll.name}_forces']
            batch = graph.batch[torch.where(nb_forces.abs() > skip_nb_threshold)[0]]
            for b in set(batch.tolist()):
                mask = torch.where(graph.batch == b, 0., 1.)
                forces *= mask.unsqueeze(-1).unsqueeze(-1)
        loss = soft_mse(forces, max_val=max_val)

    elif '_MSPE' in loss_type.name:
        train_term = getattr(ParamTerm, loss_type.name[:-len('_MSPE')])
        topo_params = set_trainable_terms(topo_params, [train_term], max_config)
        label = getattr(graph, train_term.name + "_label")
        param_range = kwargs.pop('param_range', {})
        if train_term.name in param_range:
            label = torch.clamp(label, *param_range[train_term.name])
        loss = torch.mean(((topo_params[train_term] - label) / label)**2)

    elif '_MSE' in loss_type.name:
        train_term = getattr(ParamTerm, loss_type.name[:-len('_MSE')])
        topo_params = set_trainable_terms(topo_params, [train_term], max_config)
        label = getattr(graph, train_term.name + "_label")
        param_range = kwargs.pop('param_range', {})
        if train_term.name in param_range:
            label = torch.clamp(label, *param_range[train_term.name])
        loss = torch.mean(((topo_params[train_term] - label))**2)

    else:
        raise NotImplementedError(loss_type)

    return loss


class EnergyForceLossType(Enum):
    Energy_MSE = 1
    Energy_Soft_MSE = 2
    Force_MSE = 3
    Force_Soft_MSE = 4
    L1_Norm = 5


def energyforce_loss(topo_params: TopoParams,
                     graph: MolData,
                     loss_type: EnergyForceLossType,
                     max_config: dict[str, float] = None,
                     **kwargs):

    # print(topo_params[ParamTerm.ProperTorsion_k][:4])

    topo_params = set_trainable_terms(topo_params, [
        ParamTerm.Bond_k, ParamTerm.Angle_k, ParamTerm.ProperTorsion_k, ParamTerm.ImproperTorsion_k,
        ParamTerm.Bond_length, ParamTerm.Angle_theta
    ], max_config)

    if loss_type is EnergyForceLossType.Energy_MSE:

        confmask = graph.confmask
        results = CFF.energy_force(
            topo_params,
            graph.coords,
            calc_force=False,
            confmask=confmask,
        )
        energy = results['energy']
        energy_label = graph.label_energy
        shift = calc_conf_mean(energy - energy_label, confmask).unsqueeze(-1)
        loss = torch.mean(calc_conf_mean((energy - energy_label - shift)**2, confmask))

    elif loss_type is EnergyForceLossType.Energy_Soft_MSE:

        max_val = kwargs.pop("max", 5.)
        confmask = graph.confmask
        results = CFF.energy_force(
            topo_params,
            graph.coords,
            calc_force=False,
            confmask=confmask,
        )
        energy = results['energy']
        energy_label = graph.label_energy
        shift = calc_conf_mean(energy - energy_label, confmask).unsqueeze(-1)

        e_loss = soft_mse(energy - energy_label - shift, max_val, keep_dim=True)
        loss = torch.mean(calc_conf_mean(e_loss, confmask))

    elif loss_type is EnergyForceLossType.Force_MSE:

        confmask = graph.confmask
        results = CFF.energy_force(
            topo_params,
            graph.coords,
            calc_force=True,
            confmask=confmask,
        )
        forces = results['forces']
        forces_label = graph.forces

        f_loss = torch.mean((forces - forces_label)**2, dim=-1)  # [natoms, nconfs]
        confmask_na = torch.repeat_interleave(confmask, graph.nAtom, 0)
        loss = torch.mean(calc_conf_mean(f_loss, confmask_na))

    elif loss_type is EnergyForceLossType.Force_Soft_MSE:
        max_val = kwargs.pop("max", 10.)
        confmask = graph.confmask
        results = CFF.energy_force(
            topo_params,
            graph.coords,
            calc_force=True,
            confmask=confmask,
        )
        forces = results['forces']
        forces_label = graph.forces

        f_loss = torch.mean(soft_mse(forces - forces_label, max_val, keep_dim=True), dim=-1)
        confmask_na = torch.repeat_interleave(confmask, graph.nAtom, 0)
        loss = torch.mean(calc_conf_mean(f_loss, confmask_na))

    elif loss_type == EnergyForceLossType.L1_Norm:
        loss = torch.mean(torch.abs(topo_params[ParamTerm.ProperTorsion_k]))

    else:
        raise NotImplementedError(loss_type)

    return loss
