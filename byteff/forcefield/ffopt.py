# -----  ByteFF: ByteDance Force Field -----
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates

# This file is modified from https://pytorch.org/docs/stable/generated/torch.optim.LBFGS.html#torch.optim.LBFGS.

# BSD 3-Clause License

# Copyright (c) 2017, Pytorch contributors
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.

# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import logging
from functools import partial
from typing import Callable

import torch
from torch import LongTensor, Tensor
from torch_geometric.data import Data

from byteff.forcefield.ff_kernels import (_batch_dot, _batch_reduce, _batch_to_atoms)

logger = logging.getLogger(__name__)


class BatchedLBFGS:
    """Modified from torch.optim.LBFGS 
        https://pytorch.org/docs/stable/generated/torch.optim.LBFGS.html#torch.optim.LBFGS.

    The first `nbatch_dims` are flattened to be one batch dim, 
        and the other dims are flattened to be one feature dim.
    Each row in the batch dim maintains its own history and is optimized independently.

    Args:
        parameter (Tensor): the parameter to be optimized. Only one parameter is supported.
        batch (LongTensor): record batch of each atom  [natoms, nconfs]
        lr (float): learning rate (default: 1).
        max_iter (int): maximal number of iterations per optimization step (default: 20)
        tolerance_grad (float): termination tolerance on first order optimality (default: 1e-5).
        history_size (int): update history size (default: 100).
        max_step (float): maximum update size in each iter (default: None).
        H_diag_init (flat): H diag initial guess (default: 1.).
    """

    def __init__(self,
                 parameter: Tensor,
                 batch: LongTensor,
                 confmask: Tensor = None,
                 lr: int = 1,
                 max_iter: int = 20,
                 tolerance_grad: float = 1e-7,
                 history_size: int = 100,
                 max_step: float = None,
                 H_diag_init: float = 1.):
        assert max_step is None or max_step > 0

        self._param = parameter
        self.batch = batch
        self.confmask = confmask  # [bs, nconfs]
        assert list(batch.shape) == list(parameter.shape)[:2]

        self.lr = lr
        self.max_iter = max_iter
        self.tolerance_grad = tolerance_grad
        self.history_size = history_size
        self.max_step = max_step
        self.H_diag_init = H_diag_init

        self.state = {}

    def _gather_grad(self):
        ret = torch.zeros_like(self._param) if self._param is None else self._param.grad
        return ret

    def _add_grad(self, step_size, update, max_step=None, constraints_fn=None):
        p = self._param
        update *= step_size  # [natoms, nconfs, 3]

        if max_step is not None:
            # each row are rescaled by max_step
            max_update = update.abs().max(-1)[0]  # [natoms, nconfs]
            max_update = _batch_reduce(max_update, self.batch, reduce='amax')
            scale = torch.where(max_update > max_step, max_step / max_update, 1.0)
            update *= _batch_to_atoms(scale, self.batch)

        if constraints_fn is not None:
            update, constraint_flag = constraints_fn(p, update)
        else:
            constraint_flag = torch.ones((self.batch.max() + 1, self.batch.shape[1]), dtype=torch.bool, device=p.device)

        p.add_(update)
        # return the update actually used
        return constraint_flag, update

    @torch.no_grad()
    def step(self, closure, constraints_fn=None):
        """Performs a single optimization step.

        `converge_flag` records whether a row meets convergence conditions.
        `constraint_flag` records whether a row's constraint converged.
        One row will be updated if (not converge_flag) and constraint_flag.
        Stop iteration if every row meets converge_flag or (not constraint_flag)

        Args:
            closure (Callable): A closure that reevaluates the model
                and returns the loss.
            constraints_fn (Callable): A function that adjust position update to maintain constraints
        """

        batch = self.batch
        confmask = self.confmask
        if confmask is None:
            nfake = 0
        else:
            nfake = int(confmask.nelement() - confmask.sum())
        bs = torch.max(self.batch) + 1
        nconfs = self._param.shape[1]
        lr = self.lr
        max_iter = self.max_iter
        tolerance_grad = self.tolerance_grad
        history_size = self.history_size
        max_step = self.max_step
        H_diag_init = self.H_diag_init
        state = self.state

        # evaluate initial f(x) and df/dx
        closure()
        grad = self._gather_grad()  # [nrows, nfeats]
        max_grad = _batch_reduce(grad.abs().max(dim=-1)[0], batch, reduce='amax')
        converge_flag: torch.BoolTensor = max_grad <= tolerance_grad
        if confmask is not None:
            converge_flag.logical_or_(confmask < 0.5)
        constraint_flag = torch.ones_like(converge_flag, dtype=torch.bool)

        # optimal condition
        if converge_flag.all():
            return converge_flag

        # tensors cached in state (for tracing)
        d = state.get('d')
        s = state.get('s')
        old_dirs = state.get('old_dirs')
        old_stps = state.get('old_stps')
        ro = state.get('ro')
        H_diag = state.get('H_diag')
        prev_grad = state.get('prev_flat_grad')

        n_iter = 0
        state['n_iter'] = n_iter
        # optimize for a max of max_iter iterations
        while n_iter < max_iter:
            # keep track of nb of iterations
            n_iter += 1
            state['n_iter'] = n_iter
            logger.debug("niter %d", n_iter)

            ############################################################
            # compute gradient descent direction
            ############################################################
            if n_iter == 1:
                d = grad.neg()
                ro = torch.zeros((history_size, bs, nconfs), dtype=d.dtype, device=d.device)  # [n_history, bs, nconfs]
                # [n_history, natoms, nconfs, 3]
                old_dirs = torch.zeros([history_size] + list(self._param.shape), dtype=d.dtype, device=d.device)
                old_stps = torch.zeros([history_size] + list(self._param.shape), dtype=d.dtype, device=d.device)
                H_diag = H_diag_init * torch.ones((bs, nconfs), dtype=d.dtype, device=d.device)
            else:
                # do lbfgs update (update memory)
                y = grad.sub(prev_grad)  # [natoms, nconfs, 3]
                cf_batch = _batch_to_atoms(converge_flag, batch)
                y += cf_batch.to(y.dtype)  # to aviod nan
                ys = _batch_dot(y, s, batch)  # [bs, nconfs]
                ys += converge_flag.to(y.dtype)  # to aviod nan

                # rows with ys > torch.finfo().eps * 10 is updated
                mask = ys < (torch.finfo().eps * 10)  # [bs, nconfs]
                mask_atoms = _batch_to_atoms(mask, batch)  # [natoms, nconfs, 3]
                H_diag_new = (ys / _batch_dot(y, y, batch)).nan_to_num()
                H_diag = H_diag_new * (1. - mask.to(ys.dtype)) + H_diag * mask.to(ys.dtype)  # [bs, nconfs]

                new_ro = (1. / ys).nan_to_num()
                new_dirs = y.nan_to_num()
                new_steps = s.nan_to_num()
                new_ro[mask] = 0.  # filling with 0 equals no update
                new_dirs[mask_atoms] = 0.
                new_steps[mask_atoms] = 0.

                new_index = (n_iter - 2) % history_size
                ro[new_index] = new_ro
                old_dirs[new_index] = new_dirs
                old_stps[new_index] = new_steps

                # compute the approximate (L-BFGS) inverse Hessian multiplied by the gradient
                if 'al' not in state:
                    state['al'] = [None] * history_size
                al = state['al']

                # iteration in L-BFGS loop collapsed to use just one buffer
                if n_iter - 1 <= history_size:
                    indices = list(range(n_iter - 1))
                else:
                    begin = (n_iter - 1) % history_size
                    indices = list(range(begin, history_size)) + list(range(begin))
                q = grad.neg()
                for i in indices[::-1]:
                    al[i] = _batch_dot(old_stps[i], q, batch) * ro[i]  # [bs, nconfs]
                    q.add_(old_dirs[i] * (-_batch_to_atoms(al[i], batch)))  # [natoms, nconfs, 3]

                # multiply by initial Hessian
                # r/d is the final direction
                d = r = torch.mul(q, _batch_to_atoms(H_diag, batch))
                for i in indices:
                    be_i = _batch_dot(old_dirs[i], r, batch) * ro[i]  # [bs, nconfs]
                    r.add_(old_stps[i] * _batch_to_atoms(al[i] - be_i, batch))  # [natoms, nconfs, 3]

            if prev_grad is None:
                prev_grad = grad.clone(memory_format=torch.contiguous_format)
            else:
                prev_grad.copy_(grad)

            ############################################################
            # compute step length
            ############################################################

            # update unconverged and constrainted rows
            mask = converge_flag.logical_not().logical_and(constraint_flag).to(d.dtype)
            # perform constrainted update
            mask = _batch_to_atoms(mask, batch)
            cf, s = self._add_grad(lr, d * mask, max_step=max_step, constraints_fn=constraints_fn)
            constraint_flag.logical_and_(cf)

            # re-evaluate function only if not in last iteration
            # the reason we do this: in a stochastic setting,
            # no use to re-evaluate that function here
            closure()
            grad = self._gather_grad()  # [natoms, nconfs, 3]
            max_grad = _batch_reduce(grad.abs().max(dim=-1)[0], batch, reduce='amax')  # [bs, nconfs]
            converge_flag = max_grad <= tolerance_grad
            if confmask is not None:
                converge_flag.logical_or_(confmask < 0.5)
            num, converged = converge_flag.nelement(), converge_flag.to(torch.int64).sum()
            logger.debug("FFopt not converged: %d / %d", num - converged, num - nfake)

            ############################################################
            # check conditions
            ############################################################
            if n_iter == max_iter:
                break

            # stop iteration if converged or constraint failed
            if converge_flag.logical_or(constraint_flag.logical_not()).all():
                break

        state['lbfgs_converge_flag'] = converge_flag.logical_or(constraint_flag.logical_not())
        state['constraint_converge_flag'] = constraint_flag

        if not constraint_flag.all():
            num, converged = constraint_flag.nelement(), constraint_flag.long().sum()
            logger.warning("Constraints not converged: %d / %d", num - converged, num - nfake)

        # exclude not constraint
        flag = converge_flag.logical_or(constraint_flag.logical_not())
        if not flag.all():
            num, converged = flag.nelement(), flag.long().sum()
            logger.warning("BFGS not converged: %d / %d", num - converged, num - nfake)

        return converge_flag.logical_and_(constraint_flag)


class ConstraintFFopt:

    @staticmethod
    def adjust_grad(grad: Tensor, jac: Tensor, batch: Tensor) -> Tensor:
        assert grad.shape == jac.shape
        # normalize jacobian vec
        jac_norm = torch.sqrt(_batch_dot(jac, jac, batch=batch, keep_dim=True))
        jac_unit = jac / jac_norm
        # project grad to jacobian
        proj = _batch_dot(grad, jac_unit, batch=batch, keep_dim=True)
        new_grad = grad - jac_unit * proj
        return new_grad

    @staticmethod
    def adjust_position(coords: Tensor,
                        update: Tensor,
                        batch: Tensor,
                        jacobian_func,
                        max_iter=50,
                        threshold=1e-5) -> tuple[Tensor, Tensor]:
        assert coords.shape == update.shape
        init_phi, init_jac = jacobian_func(coords)
        tmp_coords = coords.clone().detach() + update

        for _ in range(max_iter):
            phi, jac = jacobian_func(tmp_coords)
            diff = (phi - init_phi + torch.pi) % (2 * torch.pi) - torch.pi
            converge_flag = torch.abs(diff) < threshold  # [bs, nconfs]
            if converge_flag.all():
                break
            cflag = converge_flag.to(coords.dtype)
            # [natoms, nconfs, 3]
            cflag = _batch_to_atoms(cflag, batch)
            mask = 1. - cflag
            inner = _batch_dot(init_jac, jac, batch=batch, keep_dim=True)
            diff = _batch_to_atoms(diff, batch)
            tmp_coords -= diff * init_jac / inner * mask

        if not converge_flag.all():
            tmp_coords[cflag < 1.] = coords[cflag < 1.]

        return tmp_coords - coords, converge_flag

    @classmethod
    @torch.no_grad()
    def optimize(cls,
                 graph: Data,
                 *,
                 energy_func: Callable,
                 jacobian_func: Callable,
                 pos_res_k: float = 0.,
                 max_iter: int = 1000,
                 max_step: float = 0.1,
                 history_size: int = 100,
                 f_max: float = 0.01) -> tuple[Tensor, Tensor]:
        """Optimize batched coordinates with constraint and position restraint.
        graph (Data): contraining coords and other informations
        energy_func (Callable): f(graph) -> energy ([batch_size, nconfs]), force (coords.shape),
        jacobian_func (Callable): f(graph) -> 
            constraint value ([batch_size, nconfs]), constraint jacobian (coords.shape)
        """
        coords = graph.coords.clone()  # [natoms, nconfs, 3]
        init_coords = coords.clone()
        init_cons_value = jacobian_func(coords)[0]
        batch = graph.batch.clone().unsqueeze(-1).expand(-1, coords.shape[1])  # [bs, nconfs]
        optimizer = BatchedLBFGS(coords,
                                 batch=batch,
                                 lr=1.0,
                                 max_iter=max_iter,
                                 max_step=max_step,
                                 tolerance_grad=f_max,
                                 history_size=history_size,
                                 H_diag_init=1. / 70.)

        def closure():

            energy, force = energy_func(coords)
            # position restraint
            pe = (0.5 * pos_res_k * (coords - init_coords)**2).sum(-1)
            energy += _batch_reduce(pe, batch)
            pe_force = -pos_res_k * (coords - init_coords)
            force += pe_force

            # project force
            _, jac = jacobian_func(coords)
            new_force = cls.adjust_grad(force, jac, batch)
            coords.grad = -new_force
            loss = torch.sum(energy)
            return loss

        converge_flag = optimizer.step(closure, partial(cls.adjust_position, jacobian_func=jacobian_func, batch=batch))
        assert optimizer.state['n_iter'] > 1, "LBFGS failed"
        logger.info(f"ffopt lbfgs niter: {optimizer.state['n_iter']}")

        dc_max = torch.max(torch.abs(coords - init_coords))
        if dc_max > 2.0:
            logger.warning(f"large coords diff: {dc_max}")

        cons_value = jacobian_func(coords)[0]
        dd_max = torch.max((cons_value - init_cons_value + torch.pi) % (2 * torch.pi) - torch.pi)
        if dd_max > 1e-2:
            logger.warning(f"large constraint diff: {dd_max}")

        graph.coords = init_coords
        return coords, converge_flag
