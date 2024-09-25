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
"""
    implementation of SWAG

    Ref:
    https://arxiv.org/pdf/1902.02476.pdf (A Simple Baseline for Bayesian Uncertainty in Deep Learning)
    https://github.com/wjmaddox/swa_gaussian (github)
    https://arxiv.org/pdf/2109.08248.pdf (Assessments of epistemic uncertainty using Gaussian stochastic weight averaging for fluid-flow regression)

"""

import copy

import torch
from torch import Tensor
from torch.nn import Module


def flatten(lst: list[Tensor]):
    tmp = [i.contiguous().view(-1, 1) for i in lst]
    return torch.cat(tmp).view(-1)


def unflatten_like(vector: Tensor, likeTensorList: list[Tensor]):
    # Takes a flat torch.tensor and unflattens it to a list of torch.tensors
    #    shaped like likeTensorList
    outList = []
    i = 0
    for tensor in likeTensorList:
        # n = module._parameters[name].numel()
        n = tensor.numel()
        outList.append(vector[:, i:i + n].view(tensor.shape))
        i += n
    return outList


class SWAG(torch.nn.Module):

    def __init__(self, base: torch.nn.Module, use_cov_mat=True, max_num_models=20, var_clamp=1e-30):
        super(SWAG, self).__init__()

        self.n_models = 0
        self.params = list()

        self.use_cov_mat = use_cov_mat
        self.max_num_models = max_num_models

        self.var_clamp = var_clamp

        # only used for validation
        self.base = copy.deepcopy(base)

        # save states on cpu
        self.mean: dict[str, Tensor] = None
        self.sq_mean: dict[str, Tensor] = None
        self.cov_mat_sqrt: dict[str, Tensor] = None

    def forward(self, *args, **kwargs):
        return self.base(*args, **kwargs)

    def collect_model(self, base_model: Module):
        if self.mean is None:
            print("called")
            # the first time updating parameters in SWAG
            self.mean = {n: p.clone().detach().cpu() for n, p in base_model.named_parameters()}
            self.sq_mean = {n: p.clone().detach().pow(2).cpu() for n, p in base_model.named_parameters()}
            if self.use_cov_mat:
                self.cov_mat_sqrt = {
                    n: torch.zeros((0, p.data.numel()), dtype=p.dtype) for n, p in base_model.named_parameters()
                }
        else:
            c = self.n_models
            for n, p in base_model.named_parameters():
                self.mean[n] = (self.mean[n] * c + p.detach().cpu()) / (c + 1)
                self.sq_mean[n] = (self.sq_mean[n] * c + p.detach().pow(2).cpu()) / (c + 1)
                if self.use_cov_mat:
                    dif = (p.detach().cpu() - self.mean[n]).view(-1, 1)  # [N_param, 1]
                    self.cov_mat_sqrt[n] = torch.cat((self.cov_mat_sqrt[n], dif.t()), dim=0)  # [N_epoch, N_param]
                    if self.cov_mat_sqrt[n].shape[0] + 1 > self.max_num_models:
                        # store max_num_models, discard the first saved model
                        self.cov_mat_sqrt[n] = self.cov_mat_sqrt[n][1:, :]

        self.n_models += 1

    def sample(self, scale=1.0, cov=False, seed=None):
        if seed is not None:
            torch.manual_seed(seed)

        scale_sqrt = scale**0.5

        mean_list = []
        sq_mean_list = []
        names = []

        if cov:
            cov_mat_sqrt_list = []

        for n in self.mean:
            mean = self.mean[n]
            sq_mean = self.sq_mean[n]

            if cov:
                cov_mat_sqrt = self.cov_mat_sqrt[n]
                cov_mat_sqrt_list.append(cov_mat_sqrt)

            names.append(n)
            mean_list.append(mean)
            sq_mean_list.append(sq_mean)

        mean = flatten(mean_list)
        sq_mean = flatten(sq_mean_list)

        # draw diagonal variance sample
        var = torch.clamp(sq_mean - mean**2, self.var_clamp)
        var_sample = var.sqrt() * torch.randn_like(var, requires_grad=False)

        # if covariance draw low rank sample
        if cov and self.n_models > 1:
            cov_mat_sqrt = torch.cat(cov_mat_sqrt_list, dim=1)

            cov_sample = cov_mat_sqrt.t().matmul(
                cov_mat_sqrt.new_empty((cov_mat_sqrt.size(0),), requires_grad=False).normal_())
            cov_sample /= (min(self.n_models, self.max_num_models) - 1)**0.5

            rand_sample = var_sample + cov_sample
        else:
            rand_sample = var_sample

        # update sample with mean and scale
        sample = mean + scale_sqrt * rand_sample
        sample = sample.unsqueeze(0)

        # unflatten new sample like the mean sample
        samples_list = unflatten_like(sample, mean_list)
        sample_dict = {n: p for n, p in zip(names, samples_list)}

        for n, p in self.base.named_parameters():
            p.data.copy_(sample_dict[n].to(p.device))

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        state = dict(super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars))
        state['n_models'] = self.n_models
        state["mean"] = self.mean
        state["sq_mean"] = self.sq_mean
        state["cov_mat_sqrt"] = self.cov_mat_sqrt
        return state

    def load_state_dict(self, state_dict, strict=True):
        self.n_models = state_dict.pop('n_models')
        self.mean = state_dict.pop("mean")
        self.sq_mean = state_dict.pop("sq_mean")
        self.cov_mat_sqrt = state_dict.pop("cov_mat_sqrt")
        super().load_state_dict(state_dict, strict)
