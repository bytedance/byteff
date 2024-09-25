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

import json
import math
import os
import random
import textwrap
from collections import OrderedDict
from copy import deepcopy
from datetime import datetime
from enum import IntEnum
from glob import glob
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.swa_utils import SWALR
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

from byteff.data import MolDataset
from byteff.forcefield import CFF
from byteff.forcefield.ff_kernels import _batch_to_atoms, dihedral_jacobian
from byteff.forcefield.ffopt import ConstraintFFopt
from byteff.model import MLParams
from byteff.model.swag import SWAG
from byteff.train import loss as loss_funcs
from byteff.utils import get_data_file_path, setup_default_logging


def safe_barrier():
    if dist.is_initialized():
        return dist.barrier()


class TrainState(IntEnum):
    NULL = 0
    STARTED = 1
    FINISHED = 2


class TrainConfig:

    def __init__(self, config: Union[str, dict] = None, timestamp=True, make_working_dir=True, restart=False):
        with open(get_data_file_path("train_config_template.yaml", "byteff.train")) as file:
            self._config: dict[str, dict] = yaml.safe_load(file)

        custom_config: dict[str, dict] = None

        if isinstance(config, dict):
            custom_config = config
        elif isinstance(config, str):
            with open(config) as file:
                custom_config = yaml.safe_load(file)
        elif config is not None:
            raise TypeError(f"Type {type(config)} is not allowed.")

        if custom_config is not None:
            for k in self._config:
                if k in custom_config:
                    if k == 'dataset':
                        defaul_config: dict = self._config[k][0]
                        for i in range(len(custom_config[k])):
                            (new_config := deepcopy(defaul_config)).update(custom_config[k][i])
                            custom_config[k][i] = new_config
                        self._config[k] = custom_config[k]
                    else:
                        self._config[k].update(custom_config[k])

        self.meta: dict = self._config['meta']
        self.dataset: list[dict] = self._config['dataset']
        self.model: dict = self._config['model']
        self.training: dict = self._config['training']

        self.work_folder = self.meta['work_folder']
        if timestamp:
            self.work_folder = self.work_folder.rstrip("/") + '_' + self.get_current_time_str()
        self.ckpt_folder = os.path.join(self.work_folder, "ckpt")
        if make_working_dir:
            assert restart or not os.path.exists(self.work_folder), self.work_folder
            os.makedirs(self.ckpt_folder, exist_ok=True)
            self.to_yaml()

    def to_yaml(self, save_path: str = None):
        if save_path is None:
            save_path = os.path.join(self.work_folder, 'fftrainer_config_in_use.yaml')
        else:
            assert save_path.endswith('.yaml')
        with open(save_path, 'w') as file:
            yaml.dump(self._config, file)

    def get_current_time_str(self):
        return datetime.now().strftime("%y_%m_%d_%H_%M_%S")

    @property
    def finish_flag(self):
        return os.path.join(self.work_folder, 'FINISHED')

    def optimal_path(self, label='') -> str:
        if label:
            return os.path.join(self.work_folder, f'optimal_{label}.pt')
        else:
            return os.path.join(self.work_folder, 'optimal.pt')

    @property
    def train_state(self) -> TrainState:
        if not os.path.exists(self.work_folder) or not os.path.exists(self.optimal_path()):
            return TrainState.NULL

        if os.path.exists(self.finish_flag):
            return TrainState.FINISHED

        else:
            return TrainState.STARTED


class ParamsTrainer:

    def start_ddp(self, rank, world_size, device, find_unused_parameters=False, restart=False):
        self.rank = rank
        self.world_size = world_size
        self.device = device

        self.logger = self._init_logger()
        if device != 'cpu':
            torch.cuda.set_device(device)

        dtype = torch.float64 if self.config.meta['fp64'] else torch.float32
        torch.set_default_dtype(dtype)
        if self.rank == 0:
            self.logger.info('set default dtype to %s', dtype)

        self._set_seed(self.config.meta['random_seed'])

        if self.load_data:
            if self.rank == 0:
                self.logger.info("loading dataset")
            self.datasets, self.train_dls, self.valid_dls = self._load_data(self.config.dataset)
        else:
            self.datasets, self.train_dls, self.valid_dls = [], [], []

        if self.rank == 0:
            self.logger.info("loading model")
        ckpt = self.config.model.pop("check_point", None)
        self.model = MLParams(**self.config.model)
        self.model.to(self.device)

        if world_size > 1:
            # set find_unused_parameters = True when some model parameters a not used in loss
            self.model = DDP(self.model, find_unused_parameters=find_unused_parameters)

        if restart:
            paths = glob(os.path.join(self.config.ckpt_folder, "ckpt_epoch_*.pt"))
            if paths:
                self._init_optimizer_scheduler()
                epochs = [int(fp.split('_')[-1].split('.')[0]) for fp in paths]
                self.load_ckpt(os.path.join(self.config.ckpt_folder, f"ckpt_epoch_{max(epochs)}.pt"), model_only=False)
                if self.rank == 0:
                    self.logger.info(f'restarted from epoch {self.epoch}')
                self.restarted = True

        if ckpt is not None and not self.restarted:
            if self.rank == 0:
                self.logger.info("loading check point from %s", ckpt)
            self.load_ckpt(ckpt, model_only=True)

    def __init__(self,
                 config: Union[str, dict],
                 timestamp=True,
                 ddp=False,
                 device='cuda',
                 load_data=True,
                 make_working_dir=True,
                 restart=False) -> None:
        self.rank = 0
        self.world_size = 1
        self.config = TrainConfig(config, timestamp, make_working_dir=make_working_dir, restart=restart)
        self.write_log_file = make_working_dir
        self.logger = self._init_logger()
        self.model: MLParams = None
        self.optimizer: optim.Optimizer = None
        self.scheduler: optim.lr_scheduler._LRScheduler = None
        self.optimal_state_dict = None
        self.early_stop_count = 0

        self.load_data = load_data
        self.datasets, self.train_dls, self.valid_dls = [], [], []
        self.epoch_step_num = 0

        # training states
        self.epoch = 0
        self.ffopt_iter = 0
        self.ffopt_begin_epoch = [0]
        self.best_valid_loss = torch.finfo().max
        self.early_stop_count = 0
        self.train_history = [[] for _ in self.config.dataset]
        self.valid_history = [[] for _ in self.config.dataset]
        self.aux_history = [[] for _ in self.config.dataset]
        self.trainer_state_variables = [
            'epoch', 'ffopt_iter', 'ffopt_begin_epoch', 'best_valid_loss', 'early_stop_count', 'train_history',
            'valid_history', 'aux_history'
        ]
        self.restarted = False

        # swag
        self.swag_train = False
        self.swag_model: SWAG = None

        if not ddp:
            self.device = torch.device('cuda', 0) if device == 'cuda' else device
            self.start_ddp(self.rank, self.world_size, self.device, restart=restart)

    @staticmethod
    def _set_seed(seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    def _init_optimizer_scheduler(self):
        optim_config = self.config.training['optimizer'].copy()
        optim_type = optim_config.pop('type')
        if self.rank == 0:
            self.logger.info(f"initiating optimizer, lr: {optim_config['lr']:.3e}")
        self.optimizer = getattr(optim, optim_type)(self.model.parameters(), **optim_config)

        sched_config = self.config.training.get('scheduler', None)
        if sched_config is not None:
            sched_config = sched_config.copy()
            sched_type = sched_config.pop('type')
            self.scheduler = getattr(optim.lr_scheduler, sched_type)(self.optimizer, **sched_config)

    def _init_logger(self):
        """set logging config for a logger"""
        log_path = os.path.join(self.config.work_folder, 'fftrainer.log') if self.write_log_file else None
        logger = setup_default_logging(stdout=True, file_path=log_path)
        if self.rank == 0:
            logger.info(f"writing logs to {log_path}")
        return logger

    def _train_valid_split(self, dataset: MolDataset, config: dict):
        if data_nums := config.get('data_num', None):
            dataset = dataset[:data_nums]
        if shuffle := config.get('shuffle', True):
            seed = self.config.meta.get('dataset_seed', self.config.meta['random_seed'])
            self._set_seed(seed)
            dataset = dataset.shuffle()
        train_ratio = config['train_ratio']
        assert isinstance(train_ratio, float) and 0. < train_ratio < 1.
        train_num = round(len(dataset) * train_ratio)
        train_ds, valid_ds = dataset[:train_num], dataset[train_num:]
        train_dl = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=shuffle, drop_last=True)
        valid_dl = DataLoader(valid_ds, batch_size=config['batch_size'], shuffle=False, drop_last=False)
        return train_dl, valid_dl, int(len(train_ds) / config['batch_size'])

    def _load_data(self, dataset_config: list[dict]) -> tuple[list[MolDataset], list[DataLoader], list[DataLoader]]:
        datasets = []
        train_dls, valid_dls, epoch_steps = [], [], []
        for i, config in enumerate(dataset_config):

            if 'valid_root' in config:
                # only used in finetuning
                train_dataset = MolDataset(root=config['root'],
                                           rank=self.rank,
                                           world_size=self.world_size,
                                           shard_id=config.get('shard_id', None),
                                           shards=config.get('shards', 1),
                                           save_label=config.get('save_label', ''))
                train_dl = DataLoader(train_dataset,
                                      batch_size=config['batch_size'],
                                      shuffle=config.get('shuffle', True),
                                      drop_last=True)
                valid_dataset = MolDataset(root=config['valid_root'],
                                           rank=self.rank,
                                           world_size=self.world_size,
                                           shard_id=None,
                                           shards=config.get('shards', 1),
                                           save_label=config.get('save_label', ''))
                valid_dl = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=False, drop_last=False)
                datasets.append(train_dataset)
                train_dls.append(train_dl)
                valid_dls.append(valid_dl)
                epoch_steps.append(int(len(train_dataset) / config['batch_size']))
                self.logger.info(f"dataset {i}, num data: {len(train_dataset)+len(valid_dataset)}")
            else:
                dataset = MolDataset(root=config['root'],
                                     rank=self.rank,
                                     world_size=self.world_size,
                                     shard_id=config.get('shard_id', None),
                                     shards=config.get('shards', 1),
                                     save_label=config.get('save_label', ''))
                datasets.append(dataset)
                train_dl, valid_dl, train_step = self._train_valid_split(dataset, config)
                train_dls.append(train_dl)
                valid_dls.append(valid_dl)
                epoch_steps.append(train_step)
                self.logger.info(f"dataset {i}, num data: {len(dataset)}")
        min_step = torch.tensor(min(epoch_steps), dtype=torch.int32, device=self.device)
        if self.world_size > 1:
            dist.all_reduce(min_step, op=dist.ReduceOp.MIN)
        self.epoch_step_num = min_step.item()
        return datasets, train_dls, valid_dls

    def save_ckpt(self, save_path: str = None, debug: bool = False):
        if self.rank == 0 or debug:
            if save_path is None:
                ckpt_savepath = os.path.join(self.config.ckpt_folder, f"ckpt_epoch_{self.epoch}.pt")
            else:
                ckpt_savepath = save_path

            self.logger.info(f'saving ckpt to: {ckpt_savepath}')
            sd = {
                'model_state_dict': self.model.module.state_dict() if self.world_size > 1 else self.model.state_dict(),
                'optimal_state_dict': self.optimal_state_dict,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler is not None else None,
                'trainer_state_dict': {
                    name: getattr(self, name) for name in self.trainer_state_variables
                },
            }
            if self.swag_train:
                sd['swag_state_dict'] = self.swag_model.state_dict()
            torch.save(sd, ckpt_savepath)

    def load_ckpt(self, ckpt_path: str, model_only=True):
        sd = torch.load(ckpt_path, map_location=self.device)
        model = self.model.module if self.world_size > 1 else self.model
        try:
            model.load_state_dict(sd['model_state_dict'])
        except RuntimeError as e:
            if self.rank == 0:
                self.logger.warning("%s", e)
                self.logger.warning("load state dict failed, use strict=False")
            new_sd = self.model.module.state_dict() if self.world_size > 1 else self.model.state_dict()
            old_sd = sd['model_state_dict']
            for k in list(old_sd.keys()):
                if k in new_sd and new_sd[k].shape != old_sd[k].shape:
                    old_sd.pop(k)
            model.load_state_dict(old_sd, strict=False)
        if model_only:
            return

        self.optimal_state_dict = sd['optimal_state_dict']
        self.optimizer.load_state_dict(sd['optimizer_state_dict'])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(sd['scheduler_state_dict'])

        for k in self.trainer_state_variables:
            setattr(self, k, sd['trainer_state_dict'][k])

        if self.swag_train:
            self.swag_model.load_state_dict(sd['swag_state_dict'])

    def save_optimal(self):
        optimal_state_dict: dict[str, torch.Tensor] = self.model.module.state_dict().copy(
        ) if self.world_size > 1 else self.model.state_dict().copy()
        self.optimal_state_dict = OrderedDict()
        for k, v in optimal_state_dict.items():
            self.optimal_state_dict[k] = v.clone().detach()
        if self.rank == 0:
            torch.save({'model_state_dict': self.optimal_state_dict}, self.config.optimal_path())

    def load_optimal(self):
        if self.optimal_state_dict is not None:
            if self.world_size > 1:
                self.model.module.load_state_dict(self.optimal_state_dict)
            else:
                self.model.load_state_dict(self.optimal_state_dict)

    def calc_loss(self, pred: dict, graph: Batch, dataset_index: int, is_valid=True) -> list[torch.FloatTensor]:

        label_type = self.config.dataset[dataset_index]['label_type']
        loss_types = self.config.dataset[dataset_index]['loss']
        aux_loss_types = self.config.dataset[dataset_index].get('aux_loss', None)

        if self.config.meta.get("print_grad", False) and not is_valid:
            for v in pred.param.values():
                if v.requires_grad:
                    v.retain_grad()

        losses = [0.]
        loss_func = getattr(loss_funcs, f"{label_type.lower()}_loss")
        loss_type_class = getattr(loss_funcs, f"{label_type}LossType")

        loss_dict = {}
        for dct in loss_types:
            name = dct['loss_type']
            max_config = None if is_valid else dct.get('max_config', None)
            kwargs = dct.get('kwargs', {})
            loss_type = getattr(loss_type_class, name)
            # print(loss_type)
            loss = loss_func(pred,
                             graph,
                             loss_type=loss_type,
                             max_config=max_config,
                             param_range=self.config.model['param_range'],
                             **kwargs)
            loss_dict[name] = [kwargs, loss]
            weight = dct.get('valid_weight', dct['weight']) if is_valid else dct['weight']
            losses[0] += loss * weight

        if aux_loss_types is not None and is_valid:
            for dct in aux_loss_types:
                kwargs = dct.get('kwargs', {})
                name = dct['loss_type']
                if name in loss_dict and kwargs == loss_dict[name][0]:
                    losses.append(loss_dict[name][1])
                else:
                    losses.append(
                        loss_func(pred,
                                  graph,
                                  loss_type=getattr(loss_type_class, name),
                                  param_range=self.config.model['param_range'],
                                  **kwargs))
        return losses, pred

    @torch.no_grad()
    def ffopt(self, graph: Batch, config: dict):
        params = self.model(graph)
        torsion_index = graph.torsion_index

        def energy_func(_coords):
            graph.coords = _coords
            pred = CFF.energy_force(params, graph.coords, confmask=graph.confmask)
            return pred['energy'], pred['forces']

        def jacobian_func(_coords):
            phi, jac = dihedral_jacobian(_coords, torsion_index)
            return phi, jac

        config = config.copy()
        rk = config['pos_res_k']
        config['pos_res_k'] = rk[self.ffopt_iter] if isinstance(rk, list) else rk
        new_coords, converge_flag = ConstraintFFopt.optimize(graph,
                                                             energy_func=energy_func,
                                                             jacobian_func=jacobian_func,
                                                             **config)
        if new_coords.isnan().any():
            self.logger.warning(f"converge flag, {torch.where(torch.logical_not(converge_flag))}")
            cc = new_coords.sum(-1).sum(-1)
            self.logger.warning(f"find nan, {torch.where(cc.isnan())}")
            new_coords = torch.nan_to_num(new_coords)

        # if not converged, use init coords for all the confs of the molecule
        if not converge_flag.all():
            ids = set(torch.where(torch.logical_not(converge_flag))[0].tolist())
            for idx in ids:
                self.logger.warning(f'ffopt not converge {graph.name[idx]}')
            mask = converge_flag.all(-1, keepdim=True).expand(-1, new_coords.shape[1])
            mask = _batch_to_atoms(mask.to(new_coords.dtype), graph.batch.unsqueeze(-1).expand(-1, new_coords.shape[1]))
            new_coords = (new_coords * mask + graph.coords * (1 - mask)).detach()
        return new_coords

    def ffopt_all(self, ffopt_iter: int):
        for i, config in enumerate(self.config.dataset):
            if 'ffopt' in config:
                # skip ffopt if pos_res_k < 0.
                if config['ffopt']['pos_res_k'][self.ffopt_iter] < 0.:
                    continue
                save_label = f"ffopt_{ffopt_iter}"
                dataset = self.datasets[i]
                if self.train_dls[i]:
                    self.train_dls[i], self.valid_dls[i] = None, None  # release memory
                try:
                    dataset.load_with_label(save_label)
                    if self.rank == 0:
                        self.logger.info(f"loaded ffopt result from {dataset.processed_paths[0]}")
                except FileNotFoundError:
                    if self.rank == 0:
                        self.logger.info(f"start ffopt dataset {i}, {config['label_type']}")
                    batch_size = config.get('batch_size_ffopt', config['batch_size'])
                    dataset.load_with_label("")
                    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
                    for idx, graph in enumerate(dl):
                        graph.to(self.device)
                        coords = self.ffopt(graph, config['ffopt'])
                        dataset.update_data(coords, graph.nAtom, idx * batch_size, 'coords')
                        torch.cuda.empty_cache()
                    dataset.save_with_label(save_label)
                    del dl
                    if self.rank == 0:
                        self.logger.info(f"saved ffopt result to {dataset.processed_paths[0]}")
                self.train_dls[i], self.valid_dls[i], _ = self._train_valid_split(dataset, config)

    def train_epoch(self):
        # train goes over same steps on each dataloader_impl
        # depending on train_bs, some data may be unused
        # the train loop takes one minibatch from each train_dataset,
        # calculates the grad, saves the grad and continues to next minibatch from next dataloader_impl

        self.model.train()
        self.logger.debug('starting training epoch')
        safe_barrier()
        self.logger.debug(f'train steps: {self.epoch_step_num}, rank: {self.rank}')
        try:
            loaders = [iter(l) for l in self.train_dls]

            for step in range(self.epoch_step_num):

                total_loss = 0.
                self.optimizer.zero_grad()

                for ids, loader in enumerate(loaders):
                    graph = next(loader).to(self.device)
                    pred = self.model(graph)
                    losses, pred = self.calc_loss(pred, graph, ids, is_valid=False)
                    loss = losses[0]
                    total_loss += loss * self.config.dataset[ids]['loss_weight']
                    self.train_history[ids].append([self.epoch, step, loss.item()])
                    self.logger.info(
                        f'Train epoch {self.epoch}, step {step}, dataset {ids}, rank {self.rank}, loss: {loss.item()}')

                    if self.config.meta.get('save_debug_data', False) and self.rank == 0:
                        if len(self.train_history[ids]) > 1 and loss.item() > 1.5 * self.train_history[ids][-2][-1]:
                            save_dir = os.path.join(self.config.work_folder, "debug")
                            os.makedirs(save_dir, exist_ok=True)
                            self.save_ckpt(os.path.join(save_dir, f"ckpt_{self.epoch}_{step}.pt"), debug=True)
                            torch.save(graph, os.path.join(save_dir, f"data_{self.epoch}_{step}.pt"))
                            self.logger.info("debug data saved!")

                safe_barrier()
                total_loss.backward()

                if self.config.meta.get("print_grad", False) and self.rank == 0:
                    for k, v in pred.param.items():
                        if v.grad is not None:
                            print(k)
                            print(
                                f"grad mean: {v.grad.mean():.4e}, abs max: {v.grad.abs().max():.4e}, std: {v.grad.std():.4e}"
                            )
                        else:
                            print("None grad", k)
                    # raise
                self.optimizer.step()

        except KeyboardInterrupt:
            if self.rank == 0:
                self.logger.info('stopped by KeyboardInterrupt')
            exit(-1)

    # @torch.no_grad()
    def valid_epoch(self, use_swa=False):

        safe_barrier()
        self.model.eval()
        self.logger.debug('starting validation epoch')

        averaged_loss = 0.
        for ids, valid_dl in enumerate(self.valid_dls):

            tot_loss = 0.
            nbatch = torch.tensor(0, dtype=torch.int64, device=self.device)
            for graph in valid_dl:
                graph: Batch = graph.to(self.device)
                nbatch += graph.batch_size
                if use_swa:
                    self.swag_model.sample(0.)
                    pred = self.swag_model(graph)
                else:
                    pred = self.model(graph)
                loss, _ = self.calc_loss(pred, graph, ids, is_valid=True)
                loss = torch.tensor(loss, dtype=torch.float64, device=self.device)
                tot_loss += loss * graph.batch_size

            if self.world_size > 1:
                dist.all_reduce(tot_loss)
                dist.all_reduce(nbatch)

            losses = tot_loss / nbatch
            averaged_loss += losses[0] * self.config.dataset[ids]['loss_weight']

            losses = losses.detach().tolist()
            self.valid_history[ids].append([self.epoch, 0, losses[0]])
            if len(losses) > 1:
                self.aux_history[ids].append([self.epoch, 0] + losses[1:])

            if self.rank == 0:
                self.logger.info(f'valid epoch {self.epoch}, dataset {ids}, loss: {losses[0]}')

        if self.rank == 0:
            self.logger.info(f'valid epoch {self.epoch} combined rmse {averaged_loss}')

        return averaged_loss

    def train_and_valid(self):

        while True:

            if self.epoch % self.config.training['ckpt_interval'] == 0:
                self.save_ckpt()

            # reach max epochs for training iteration
            if (self.epoch - self.ffopt_begin_epoch[-1]) >= self.config.training['ffopt_interval']:
                break

            if self.epoch % self.config.training['valid_interval'] == 0:
                averaged_loss = self.valid_epoch()
                if self.scheduler is not None:
                    self.scheduler.step(averaged_loss)
                    if self.rank == 0:
                        self.logger.info(f"current learning rate: {self.optimizer.param_groups[0]['lr']:.3e}")

                if averaged_loss > self.best_valid_loss - self.config.training['ignore_tolerance']:
                    self.early_stop_count += 1
                else:
                    self.early_stop_count = 0
                    self.best_valid_loss = averaged_loss

                if self.rank == 0:
                    self.logger.info(f'early_stop_count: {self.early_stop_count}')

                if averaged_loss <= self.best_valid_loss:
                    self.save_optimal()

                if self.rank == 0:
                    self.plot_history()

                # early stop:
                if self.config.training['early_stop_patience'] <= self.early_stop_count:
                    if self.rank == 0:
                        self.logger.info(f"Early stop! Best combined rmse: {self.best_valid_loss}")
                    break

            self.train_epoch()
            if self.rank == 0:
                self.plot_history()

            self.epoch += 1

        return self.epoch

    def train_loop(self):
        """
        ffopt and train model iteratively
        """

        ffopt_iters = self.config.training.get('ffopt_iters', 1)

        while self.ffopt_iter < ffopt_iters:

            # init optimizer and scheduler at each ffopt iteration,
            # skip init if restarted
            if self.restarted:
                self.restarted = False
            else:
                self._init_optimizer_scheduler()

            if self.rank == 0:
                self.logger.info(f"ffopt iteration: {self.ffopt_iter}")

            self.ffopt_all(self.ffopt_iter)

            self.train_and_valid()

            # load optimal parameters to model for next ffopt
            self.load_optimal()
            self.ffopt_iter += 1
            self.ffopt_begin_epoch.append(self.epoch)
            self.best_valid_loss = torch.finfo().max
            self.early_stop_count = 0
            self.save_ckpt()

        with open(self.config.finish_flag, "w"):
            pass

    def plot_history(self):
        history = {"train": self.train_history, "valid": self.valid_history, "aux": self.aux_history}
        with open(os.path.join(self.config.work_folder, 'history.json'), "w") as file:
            json.dump(history, file)

        plt.cla()
        plt.clf()
        nds = len(self.train_dls)
        fig, axes = plt.subplots(1, nds, figsize=(4 * nds, 3), constrained_layout=True)
        axes = [axes] if nds == 1 else axes.flat

        for ids, ax in enumerate(axes):
            epoch_to_step = OrderedDict()
            epoch_to_step[0] = 0
            for i, res in enumerate(self.train_history[ids]):
                epoch = res[0]
                if epoch not in epoch_to_step:
                    epoch_to_step[epoch] = i
            epoch_to_step[max(epoch_to_step) + 1] = len(self.train_history[ids])

            step_to_epoch = OrderedDict()
            for epoch, step in epoch_to_step.items():
                step_to_epoch[step] = epoch

            train_rmse = []
            for i, (epoch, _, rmse) in enumerate(self.train_history[ids]):
                train_rmse.append([i, rmse])
            train_rmse = np.asarray([[0, np.nan]]) if len(train_rmse) == 0 else np.asarray(train_rmse)
            ax.plot(train_rmse[:, 0], train_rmse[:, 1], label='train')
            if train_rmse[:, 1].max() / train_rmse[:, 1].min() > 20.:
                ax.semilogy()

            valid_rmse = []
            for epoch, _, rmse in self.valid_history[ids]:
                valid_rmse.append([epoch_to_step[epoch], rmse])
            valid_rmse = np.asarray(valid_rmse)
            ax.plot(valid_rmse[:, 0], valid_rmse[:, 1], '.-', label='valid')

            ax.set_xlabel('step')
            ax.grid(visible=True, zorder=1)
            secax = ax.secondary_xaxis('top')
            secax.set_xlabel('epoch')
            epoch_ticks = np.asarray(sorted([(k, v) for k, v in step_to_epoch.items()]))
            if len(epoch_ticks) < 10:
                secax.set_xticks(ticks=epoch_ticks[:, 0], labels=epoch_ticks[:, 1])
            else:
                skip = len(epoch_ticks) // 10 + 1
                secax.set_xticks(ticks=epoch_ticks[::skip, 0], labels=epoch_ticks[::skip, 1])
            up, down = ax.get_ylim()
            ax.vlines([epoch_to_step[ep] for ep in self.ffopt_begin_epoch],
                      up,
                      down,
                      linestyles='dashed',
                      colors='black',
                      zorder=1.5)
            loss_str = ' '.join([f"{l['loss_type']}: {l['weight']}" for l in self.config.dataset[ids]['loss']])
            loss_str = '\n'.join(textwrap.wrap(loss_str, width=75))
            ax.set_title(f"{self.config.dataset[ids]['label_type']} \n {loss_str}", fontdict={"fontsize": 8})
            ax.legend(frameon=False, fontsize="small")
            ax.set_ylim(up, down)

            if 'aux_loss' in self.config.dataset[ids] and self.config.dataset[ids]['aux_loss']:
                nax = len(self.config.dataset[ids]['aux_loss'])
                nrows = math.ceil(nax / 3)
                aux_fig, aux_axes = plt.subplots(nrows, 3, figsize=(4 * 3, 3 * nrows), constrained_layout=True)
                aux_axes = [aux_axes] if nax == 1 else aux_axes.flat
                for ida, aux_ax in enumerate(aux_axes):
                    if ida >= nax:
                        aux_ax.axis('off')
                    else:
                        aux_ax.set_title(self.config.dataset[ids]['aux_loss'][ida]['loss_type'])
                        aux_ax.set_xlabel('epoch')
                        xs, ys = [k[0] for k in self.aux_history[ids]], [k[2 + ida] for k in self.aux_history[ids]]
                        aux_ax.plot(xs, ys, 'o-')
                        if min(ys) > 0. and max(ys) / min(ys) > 20.:
                            aux_ax.semilogy()
                        up, down = aux_ax.get_ylim()
                        aux_ax.vlines(self.ffopt_begin_epoch, up, down, linestyles='dashed', colors='black', zorder=1.5)
                        aux_ax.set_ylim(up, down)

                aux_fig.suptitle(self.config.dataset[ids]["label_type"] + " auxiliary loss")
                aux_fig.savefig(os.path.join(self.config.work_folder,
                                             f'aux_history_{self.config.dataset[ids]["label_type"]}.jpg'),
                                dpi=200)
                plt.close(aux_fig)

        fig.savefig(os.path.join(self.config.work_folder, 'history.jpg'), dpi=200)
        plt.close(fig)

    def init_swag(self):

        assert 'swag' in self.config.training
        config = self.config.training['swag']
        swa_lr = config['swa_lr']
        anneal_strategy = config.get('anneal_strategy', 'linear')
        warm_up_epochs = config.get('warm_up_epochs', 10)
        anneal_epochs = config.get('anneal_epochs', 50)
        stable_epochs = config.get('stable_epochs', 10)
        sample_epochs = config.get('sample_epochs', 50)
        sample_every = config.get('sample_every', 1)
        use_cov_mat = config.get('use_cov_mat', True)
        max_num_models = config.get('max_num_models', 20)

        self.swag_train = True
        base = self.model.module if self.world_size > 1 else self.model
        self.swag_model = SWAG(base, use_cov_mat=use_cov_mat, max_num_models=max_num_models)
        self.swag_model.to(self.device)

        self._init_optimizer_scheduler()
        assert self.scheduler is None
        self.scheduler = SWALR(self.optimizer, swa_lr, anneal_epochs, anneal_strategy)

        if self.rank == 0:
            self.logger.info(
                f'SWAG schedule: warm_up_epochs {warm_up_epochs}, anneal_epochs {anneal_epochs}, stable_epochs {stable_epochs}, sample_epochs {sample_epochs}, sample_every {sample_every}'
            )
        before_sample_epochs = warm_up_epochs + anneal_epochs + stable_epochs

        return before_sample_epochs, sample_epochs, sample_every, warm_up_epochs

    def swag_loop(self):
        """
        train swag model
        """

        before_sample_epochs, sample_epochs, sample_every, warm_up_epochs = self.init_swag()

        while True:

            if self.epoch >= before_sample_epochs + sample_epochs:
                self.save_ckpt()
                break

            if self.epoch % self.config.training['ckpt_interval'] == 0:
                self.save_ckpt()

            if self.epoch >= before_sample_epochs and self.epoch % sample_every == 0:
                self.swag_model.collect_model(self.model)

            if self.epoch % self.config.training['valid_interval'] == 0:
                self.valid_epoch(use_swa=self.epoch > before_sample_epochs)

                if self.rank == 0:
                    self.plot_history()

            self.train_epoch()
            if self.rank == 0:
                self.plot_history()

            if self.epoch >= warm_up_epochs:
                self.scheduler.step()
            if self.rank == 0:
                self.logger.info(f"current learning rate: {self.optimizer.param_groups[0]['lr']:.3e}")

            self.epoch += 1

        return self.epoch
