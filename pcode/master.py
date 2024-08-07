# -*- coding: utf-8 -*-
import os
import copy
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import torch.nn.init as init
from pcode.local_training.utils import _get_noisy
from pcode.utils.stat_tracker import RuntimeTracker,AverageMeter
from pcode.local_training.fine_tuning_rep_worker import Queue,MlpModule,UniformDataset,_get_dde_output,CategoryDistributionModel
import torch
import torch.distributed as dist
import functools
import pcode.master_utils as master_utils
import pcode.cd as cd
import pcode.utils as utils
import pcode.create_coordinator as create_coordinator
import pcode.create_aggregator as create_aggregator
import pcode.create_client_sampler as create_client_sampler
import pcode.create_dataset as create_dataset
import pcode.create_metrics as create_metrics
import pcode.create_model as create_model
import pcode.utils.checkpoint as checkpoint
from pcode.utils.tensor_buffer import TensorBuffer
import pcode.utils.loss as loss
from pcode.utils.early_stopping import EarlyStoppingTracker
from pcode.utils.logging import display_perf
from pcode.datasets.partition_data import DataPartitioner


class Master(object):
    def __init__(self, conf):
        self.conf = conf
        self.graph = conf.graph
        self.logger = conf.logger
        self.random_state = conf.random_state

        # some initializations.
        self.client_ids = list(range(1, 1 + conf.n_clients))
        self.world_ids = list(range(1, 1 + conf.n_participated))
        self.is_in_childworker = False

        # define arch for master and clients.
        self._create_arch()

        # define the criterion and metrics.
        self.criterion = loss.CrossEntropyLoss(reduction="mean")
        self.metrics = create_metrics.Metrics(self.master_model, task="classification")
        self.logger.log("Master initialized model/dataset/criterion/metrics.")

        # define client sampler.
        self.client_sampler = create_client_sampler.ClientSampler(
            random_state=conf.random_state,
            logger=conf.logger,
            n_clients=conf.n_clients,
            n_participated=conf.n_participated,
            local_n_epochs=conf.local_n_epochs,
            min_local_epochs=conf.min_local_epochs,
            batch_size=conf.batch_size,
            min_batch_size=conf.min_batch_size,
        )
        self.logger.log(f"Master initialized the client_sampler.")

        # define data for training/val/test.
        self._create_data()

        # define the aggregators and coordinator.
        self.aggregator = create_aggregator.Aggregator(
            fl_aggregate=self.conf.fl_aggregate,
            model=self.master_model,
            criterion=self.criterion,
            metrics=self.metrics,
            dataset=self.fl_data_cls.dataset,
            test_loaders=self.eval_loaders,
            clientid2arch=self.clientid2arch,
            logger=self.logger,
            global_lr=self.conf.global_lr,
        )
        self.coordinator = create_coordinator.Coordinator(self.metrics)
        self.logger.log("Master initialized the aggregator/coordinator.")

        # to record the perf.
        self.perf = {
            "method": self.conf.personalization_scheme["method"],
            "round": 0,
            "global_top1": 0.0,
            "top1": 0.0,
            "corr_top1": 0.0,
            "ooc_top1": 0.0,
            "natural_shift_top1": 0.0,
            "ooc_corr_top1": 0.0,
            "mixed_top1": 0.0,
        }

        # define early_stopping_tracker.
        self.early_stopping_tracker = EarlyStoppingTracker(
            patience=conf.early_stopping_rounds
        )

        # save arguments to disk.
        self.is_finished = False
        checkpoint.save_arguments(conf)

    def _create_arch(self):
        # create master model.
        _, self.master_model = create_model.define_model(
            self.conf, to_consistent_model=False
        )
        if self.conf.arch == "simple_cnn":
            self.conf.rep_len = 64
        elif "resnet" in self.conf.arch:
            resnet_size = int(self.conf.arch.replace("resnet", ""))
            if "cifar" in self.conf.data:
                self.conf.rep_len = 64*4 if resnet_size >= 44 else 64
            elif  "oh" in self.conf.data:
                self.conf.rep_len = 2048 if resnet_size >= 44 else 512
            elif "imagenet" in self.conf.data:
                self.conf.rep_len = 256
        elif "vision_transformer" in self.conf.arch:
            if "cifar10" in self.conf.data:
                self.conf.rep_len = 64
        elif "vgg" in self.conf.arch:
            if "cifar10" in self.conf.data:
                self.conf.rep_len = 256
        elif "compact_conv_transformer" in self.conf.arch:
            if "cifar10" in self.conf.data:
                self.conf.rep_len = 128
        elif "noised_cnn" in self.conf.arch:
            self.conf.rep_len = 64
        else:
            raise NotImplementedError
        # self.conf.comm_buffer_size = self.conf.rep_len + 10

        # create client model (may have different archs, but not supported yet).
        self.used_client_archs = set(
            [
                create_model.determine_arch(
                    client_id=client_id,
                    n_clients=self.conf.n_clients,
                    arch=self.conf.arch,
                    use_complex_arch=True,
                    arch_info=self.conf.arch_info,
                )
                for client_id in range(1, 1 + self.conf.n_clients)
            ]
        )

        self.logger.log(f"The client will use archs={self.used_client_archs}.")
        self.logger.log("Master created model templates for client models.")
        self.client_models = dict(
            create_model.define_model(self.conf, to_consistent_model=False, arch=arch)
            for arch in self.used_client_archs
        )
        # add an old_client_models here for the purpose of client sampling
        self.old_client_models = dict(
            create_model.define_model(self.conf, to_consistent_model=False, arch=arch)
            for arch in self.used_client_archs
        )
        self.clientid2arch = dict(
            (
                client_id,
                create_model.determine_arch(
                    client_id=client_id,
                    n_clients=self.conf.n_clients,
                    arch=self.conf.arch,
                    use_complex_arch=True,
                    arch_info=self.conf.arch_info,
                ),
            )
            for client_id in range(1, 1 + self.conf.n_clients)
        )
        self.logger.log(
            f"Master initialize the clientid2arch mapping relations: {self.clientid2arch}."
        )

    def _create_data(self):
        # create dataset (as well as the potential data_partitioner) for training.
        dist.barrier()
        self.fl_data_cls = create_dataset.FLData(
            conf=self.conf,
            logger=self.logger,
            graph=self.graph,
            random_state=self.random_state,
            batch_size=self.conf.batch_size,
            img_resolution=self.conf.img_resolution,
            corr_seed=self.conf.corr_seed,
            corr_severity=self.conf.corr_severity,
            local_n_epochs=self.conf.local_n_epochs,
            num_workers=self.conf.num_workers,
            pin_memory=self.conf.pin_memory,
        )
        self.fl_data_cls.define_dataset(
            data_name=self.conf.data,
            data_dir=self.conf.data_dir,
            is_personalized=self.conf.is_personalized,
            test_partition_ratio=self.conf.test_partition_ratio,
            extra_arg="cifar10.1" if self.conf.data == "cifar10" else self.conf.natural_shifted_imagenet_type
        )
        self.fl_data_cls.inter_clients_data_partition(
            dataset=self.fl_data_cls.dataset["train"],
            n_clients=self.conf.n_clients,
            partition_data_conf=self.conf.partition_data_conf,
        )
        self.logger.log("Master initialized the data.")

        # create test loaders.
        # client_id starts from 1 to the # of clients.
        if self.conf.is_personalized:
            # if personalization is enabled, self.dataset["test"] becomes backup test set.
            # Then we should obtain the clients' validation or test set from merged train set.
            self.eval_loaders = {}
            list_of_local_mini_batch_size = self.client_sampler.get_n_local_mini_batchsize(
                self.client_ids
            )
            eval_datasets = []
            _create_dataloader_fn = functools.partial(
                self.fl_data_cls.create_dataloader, batch_size=list_of_local_mini_batch_size[0], shuffle=True
            )
            local_train_ratio = self.conf.local_train_ratio
            local_test_ratio = (1 - local_train_ratio) / 2
            for client_id in self.client_ids:
                data_to_load = self.fl_data_cls.data_partitioner.use(client_id - 1)
                local_data_partitioner = DataPartitioner(
                    data_to_load,
                    partition_sizes=[
                        local_train_ratio,
                        1 - (local_train_ratio + local_test_ratio),
                        local_test_ratio,
                        ],
                    partition_type="random",
                    partition_alphas=None,
                    consistent_indices=False,
                    random_state=self.random_state,
                    graph=self.graph,
                    logger=self.logger,
                )
                if self.conf.eval_dataset == "val_loader":
                    eval_datasets.append(local_data_partitioner.use(1))
                elif self.conf.eval_dataset == "test_loader":
                    eval_datasets.append(local_data_partitioner.use(2))
            self.eval_loaders = {0: _create_dataloader_fn(torch.utils.data.ConcatDataset(eval_datasets))}

        else:
            test_loader = self.fl_data_cls.create_dataloader(
                self.fl_data_cls.dataset["test"], shuffle=False
            )
            self.eval_loaders = {0: test_loader}
        self.logger.log(f"Master initialized the local test data with workers.")

    def run(self):
        
        self.client_sampler.select_clients(
            model=self.master_model,
            flatten_local_models=None,
            criterion=self.criterion,
            metrics=self.metrics,
        )
        

        self.comm_round = 1
       
        # initialize lambda for drfa, need to be moved elsewhere in the future.
        if "drfa" in self.conf.personalization_scheme["method"]:
            self.drfa_lambda = np.array([1/self.conf.n_clients] * self.conf.n_clients)

        # start training.
        for comm_round in range(1, 1 + self.conf.n_comm_rounds):
            self.comm_round = comm_round
            self.logger.log(
                f"Master starting one round of federated learning: (comm_round={comm_round})."
            )

            # detect early stopping.
            self._check_early_stopping()

          

            # manually enforce client sampling on master.
            if "drfa" in self.conf.personalization_scheme["method"]:
                # Define online clients for the current round of communication for Federated Learning setting
                master_selected_lambda_idx = self.random_state.choice(
                    self.client_ids, self.conf.n_master_sampled_clients, replace=False
                ).tolist()  # for sampling the lambda
                master_selected_lambda_idx.sort()
                self.logger.log(
                        f"Sanity Check (random): Master sampled lambda idxs are: {master_selected_lambda_idx}."
                )
                self.master_selected_clients_idx = self.random_state.choice(self.client_ids, self.conf.n_master_sampled_clients, replace=False, p=self.drfa_lambda).tolist()
                self.master_selected_clients_idx.sort()
                self.logger.log(
                        f"DRFA: Master sampled client idxs are: {self.master_selected_clients_idx}."
                )
            else:
                # TODO: an more elegant way of handling this.
                self.master_selected_clients_idx = self.random_state.choice(
                    self.client_ids, self.conf.n_master_sampled_clients, replace=False
                ).tolist()
                self.master_selected_clients_idx.sort()
                self.logger.log(
                        f"Master sampled client idxs are: {self.master_selected_clients_idx}."
                )

            self._activate_selected_clients(
                self.client_sampler.selected_client_ids,
                self.client_sampler.selected_client_probs,
                self.comm_round,
                self.client_sampler.get_n_local_epoch(),
                self.client_sampler.get_n_local_mini_batchsize(),
            )

            # method-specific communications, maybe put these in different masters in the future.
            if "THE" == self.conf.personalization_scheme["method"] :
                self._send_global_rep()
            if "drfa" in self.conf.personalization_scheme["method"]:
                self._send_random_iter_index()
            
            # will decide to send the model or stop the training.
            if not self.is_finished:
                # broadcast the model to activated clients.
                self._send_model_to_selected_clients(
                    self.client_sampler.selected_client_ids
                )
                
            else:
                dist.barrier()
                self.logger.log(
                    f"Master finished the federated learning by early-stopping: (current comm_rounds={self.comm_round}, total_comm_rounds={self.conf.n_comm_rounds})"
                )
                return
            if "BTFL" in self.conf.personalization_scheme["method"]:
                self.recieve_pros(self.client_sampler.selected_client_ids)
                self.aggregation_pro(self.client_sampler.selected_client_ids)
                self._send_pro(self.client_sampler.selected_client_ids)
                #self._send_pros(self.client_sampler.selected_client_ids)
                flatten_local_models, extra_messages = self._receive_models_from_selected_clients(
                self.client_sampler.selected_client_ids
                )
                self.global_rep=torch.Tensor([])              
                for message in extra_messages.values():
                    self.global_rep = torch.cat((self.global_rep,torch.tensor(message[6:6+self.conf.rep_len])),dim=0)
               
                '''
                sub_tensor_size = self.global_rep_sigma.size(0) // 20
                sub_tensors = torch.split(self.global_rep_sigma, sub_tensor_size)
                sums = [sub_tensor.sum() for sub_tensor in sub_tensors]
                for i, sum_value in enumerate(sums):
                        print(f"第 {i+1} 个子张量的和为: {sum_value.item()}")
                '''
                self._send_global_rep_()
               
            # wait to receive the local models.
            flatten_local_models, extra_messages = self._receive_models_from_selected_clients(
                self.client_sampler.selected_client_ids
            )
            # mask out unselected models as way of naive client sampling.
            # need a more elegant implementation for sampling.
            flatten_local_models = {sel: flatten_local_models[sel] for sel in self.master_selected_clients_idx}

            if "drfa" in self.conf.personalization_scheme["method"]:
                # receive t' models
                t_prime_local_models = self._receive_models_from_selected_clients(
                    self.client_sampler.selected_client_ids
                )
                t_prime_local_models, _ = {sel: t_prime_local_models[sel] for sel in self.master_selected_clients_idx}
                # uniformly average local t_prime_models
                avg_t_prime_models = self._avg_over_archs(t_prime_local_models)
                avg_t_prime_model = list(avg_t_prime_models.values())[0]
                # send
                self.logger.log("Master send the averaged t_prime_models to workers.")
                for worker_rank, selected_client_id in enumerate(self.client_sampler.client_ids, start=1):
                    t_prime_model_state_dict = avg_t_prime_model.state_dict()
                    flatten_model = TensorBuffer(list(t_prime_model_state_dict.values()))
                    dist.send(tensor=flatten_model.buffer, dst=worker_rank)
                dist.barrier()
                # async to receive loss tensor from clients.
                reqs = {}
                drfa_loss_tensor = {id: torch.tensor(0.0) for id in self.client_sampler.selected_client_ids}
                for client_id, world_id in zip(self.client_sampler.selected_client_ids, self.world_ids):
                    req = dist.irecv(
                        tensor=drfa_loss_tensor[client_id], src=world_id
                    )
                    reqs[client_id] = req

                for client_id, req in reqs.items():
                    req.wait()
                dist.barrier()
                filtered_drfa_loss_tensor = torch.zeros(len(self.client_sampler.client_ids))
                for sel in master_selected_lambda_idx:
                    filtered_drfa_loss_tensor[sel - 1] = drfa_loss_tensor[sel]
                self.drfa_lambda += self.conf.drfa_lambda_lr * self.conf.drfa_sync_gap * filtered_drfa_loss_tensor.numpy()
                self.drfa_lambda = master_utils.euclidean_proj_simplex(torch.tensor(self.drfa_lambda)).numpy()
                self.conf.drfa_lambda_lr *= 0.9
                # avoid zero probability
                lambda_zeros = self.drfa_lambda <= 1e-3
                if lambda_zeros.sum() > 0:
                    self.drfa_lambda[lambda_zeros] = 1e-3
                self.drfa_lambda /= np.sum(self.drfa_lambda)
                self.drfa_lambda[-1] = max(0, 1 - np.sum(self.drfa_lambda[0:-1]))  # to avoid round error
                self.logger.log(f"Current lambdas are {self.drfa_lambda}.\n")

            # aggregate the local models and evaluate on the validation dataset.
            global_top1_perfs = self._aggregate_model_and_evaluate(flatten_local_models)
            

            # keep tracking the local performance
            self._track_perf(extra_messages=extra_messages, global_top1_perfs=global_top1_perfs)

            # in case we want to save a checkpoint of model
            self._save_checkpoint()
            self.logger.save_json()

            # evaluate the aggregated model.
            self.logger.log("Master finished one round of federated learning.\n")

        # formally stop the training (the master has finished all communication rounds).
        dist.barrier()
        self._finishing()

    def _save_checkpoint(self):
        if (
                self.conf.save_every_n_round is not None
                and self.comm_round % self.conf.save_every_n_round == 0
        ):
            torch.save(
                self.master_model.state_dict(),
                os.path.join(
                    self.conf.checkpoint_root, f"{self.conf.arch}_{self.comm_round}.pt"
                ),
            )

    def _activate_selected_clients(
            self,
            selected_client_ids,
            selected_client_probs,
            comm_round,
            list_of_local_n_epochs,
            list_of_local_mini_batch_size,
    ):
        # Activate the selected clients:
        # the first row indicates the client id,
        # the second row indicates the current_comm_round,
        # the third row indicates the expected local_n_epochs
        selected_client_ids = np.array(selected_client_ids)
        activation_msg = torch.zeros((5, len(selected_client_ids)))
        activation_msg[0, :] = torch.Tensor(selected_client_ids)
        activation_msg[1, :] = torch.Tensor(list(selected_client_probs.values()))
        activation_msg[2, :] = comm_round
        activation_msg[3, :] = torch.Tensor(list_of_local_n_epochs)
        activation_msg[4, :] = torch.Tensor(list_of_local_mini_batch_size)

        dist.broadcast(tensor=activation_msg, src=0)
        self.logger.log(f"Master activated the selected clients.")
        dist.barrier()

    def _send_global_rep(self):
        if not hasattr(self, "global_rep"):
            self.global_rep = torch.zeros((self.conf.rep_len,))
        dist.broadcast(tensor=self.global_rep, src=0)
        self.logger.log(f"Master sent global representation to the selected clients.")
        dist.barrier()
    def _send_global_rep_sigma_(self):
        if not hasattr(self, "global_rep_sigma"):
            self.global_rep_sigma = torch.ones((self.conf.rep_len*self.conf.n_clients,))
        dist.broadcast(tensor=self.global_rep_sigma, src=0)
        self.logger.log(f"Master sent global representation sigma true to the selected clients.")
        dist.barrier()
    def _send_global_rep_(self):
        if not hasattr(self, "global_rep"):
            self.global_rep = torch.zeros((self.conf.rep_len*self.conf.n_clients,))
        dist.broadcast(tensor=self.global_rep, src=0)
        self.logger.log(f"Master sent global representation sigma to the selected clients.")
        dist.barrier()

    def _send_random_iter_index(self):
        t_prime = torch.tensor(torch.randint(low=1, high=45*int(self.conf.local_n_epochs), size=(1,)).tolist() * len(self.client_sampler.client_ids)) # hard code a 'high'
        dist.broadcast(tensor=t_prime, src=0)
        self.logger.log(f"DRFA: Master sampled iteration t' {t_prime} and send to the selected clients.")
        dist.barrier()

    def _send_model_to_selected_clients(self, selected_client_ids):
        # the master_model can be large; the client_models can be small and different.
        self.logger.log("Master send the models to workers.")
        for worker_rank, selected_client_id in enumerate(selected_client_ids, start=1):
            arch = self.clientid2arch[selected_client_id]

            #self.client_models[arch].spectral_norm()

            if selected_client_id in self.master_selected_clients_idx:
                client_model_state_dict = self.client_models[arch].state_dict()
                flatten_model = TensorBuffer(list(client_model_state_dict.values()))
                dist.send(tensor=flatten_model.buffer, dst=worker_rank)
                self.logger.log(
                    f"\tMaster send the current model={arch} to process_id={worker_rank}."
                )
            else:
                client_model_state_dict = self.old_client_models[arch].state_dict()
                flatten_model = TensorBuffer(list(client_model_state_dict.values()))
                dist.send(tensor=flatten_model.buffer, dst=worker_rank)
                self.logger.log(
                    f"\tMaster send the previous model={arch} to process_id={worker_rank}."
                )
        dist.barrier()

    def _receive_models_from_selected_clients(self, selected_client_ids):
        self.logger.log("Master waits to receive the local models.")
        dist.barrier()

        # init the placeholders to recv the local models from workers.
        flatten_local_models = dict()
        extra_messages = dict()

        for selected_client_id in selected_client_ids:
            arch = self.clientid2arch[selected_client_id]
            client_tb = TensorBuffer(
                list(self.client_models[arch].state_dict().values())
            )
            message = torch.FloatTensor([0.0] * self.conf.comm_buffer_size)
            client_tb.buffer = torch.cat([torch.zeros_like(client_tb.buffer), message])
            flatten_local_models[selected_client_id] = client_tb

        # async to receive model from clients.
        reqs = {}
        for client_id, world_id in zip(selected_client_ids, self.world_ids):
            req = dist.irecv(
                tensor=flatten_local_models[client_id].buffer, src=world_id
            )
            reqs[client_id] = req

        for client_id, req in reqs.items():
            req.wait()

        for client_id in selected_client_ids:
            extra_messages[client_id] = flatten_local_models[client_id].buffer[-self.conf.comm_buffer_size:]
            flatten_local_models[client_id].buffer = flatten_local_models[
                                                         client_id
                                                     ].buffer[:-self.conf.comm_buffer_size]

        dist.barrier()
        self.logger.log("Master received all local models.")
        return flatten_local_models, extra_messages

    def _avg_over_archs(self, flatten_local_models):
        # get unique arch from this comm. round.
        archs = set(
            [
                self.clientid2arch[client_idx]
                for client_idx in flatten_local_models.keys()
            ]
        )
        assert len(archs) == 1, "we only support the case of same arch."

        # average for each arch.
        archs_fedavg_models = {}
        for arch in archs:
            # extract local_models from flatten_local_models.
            _flatten_local_models = {}
            for client_idx, flatten_local_model in flatten_local_models.items():
                if self.clientid2arch[client_idx] == arch:
                    _flatten_local_models[client_idx] = flatten_local_model

            # average corresponding local models.
            self.logger.log(
                f"Master uniformly average over {len(_flatten_local_models)} received models ({arch})."
            )
            fedavg_model = self.aggregator.aggregate(
                master_model=self.master_model,
                client_models=self.client_models,
                flatten_local_models=_flatten_local_models,
                aggregate_fn_name="_s1_federated_average" if self.conf.personalization_scheme["method"] != "GMA" else "_gma_fedavg",
                weights=self.drfa_lambda if self.conf.personalization_scheme["method"] == "drfa" else None,
            )
            archs_fedavg_models[arch] = fedavg_model
        return archs_fedavg_models

    def _aggregate_model_and_evaluate(self, flatten_local_models):
        # uniformly average local models with the same architecture.
        fedavg_models = self._avg_over_archs(flatten_local_models)
        fedavg_model = list(fedavg_models.values())[0]

        # update self.master_model in place.
        self.master_model.load_state_dict(fedavg_model.state_dict())
        # update self.client_models in place.
        for arch, _fedavg_model in fedavg_models.items():
            # add an old_client_models here for the purpose of client sampling.
            self.old_client_models[arch].load_state_dict(self.client_models[arch].state_dict())
            self.client_models[arch].load_state_dict(self.master_model.state_dict())

        # evaluate the aggregated model on the test data.
        perf = master_utils.do_validation(
            self.conf,
            self.coordinator,
            self.master_model,
            self.criterion,
            self.metrics,
            self.eval_loaders,
            split=self.conf.eval_dataset,
            label="global_model",
            comm_round=self.comm_round,
        ).dictionary["top1"]

        return perf

    def _track_perf(self, extra_messages, global_top1_perfs):
        # using the extra_message received from clients to get the ave perf for clients' local evaluations.
        # also track the perf of global model
        self.perf["round"] = self.comm_round

        if self.conf.is_personalized:
            # extract local performance from activated clients and average them.
            top1, corr_top1, ooc_top1, natural_shift_top1, ooc_corr_top1, mixed_top1 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            self.global_rep = torch.zeros((self.conf.rep_len,))
            if "BTFL" in self.conf.personalization_scheme["method"]:
                self.global_rep=torch.Tensor([])
                
                for message in extra_messages.values():
                    top1 = top1 + message[0]/len(extra_messages)
                    corr_top1 = corr_top1 + message[1]/len(extra_messages)
                    ooc_top1 = ooc_top1 + message[2]/len(extra_messages)
                    natural_shift_top1 = natural_shift_top1 + message[3]/len(extra_messages)
                    ooc_corr_top1 = ooc_corr_top1 + message[4]/len(extra_messages)
                    mixed_top1 = mixed_top1 + message[5]/len(extra_messages)
                    self.global_rep = torch.cat((self.global_rep,torch.tensor(message[6:6+self.conf.rep_len])),dim=0)
                
                
                   
                
            else:
                for message in extra_messages.values():
                    top1 = top1 + message[0]/len(extra_messages)
                    corr_top1 = corr_top1 + message[1]/len(extra_messages)
                    ooc_top1 = ooc_top1 + message[2]/len(extra_messages)
                    natural_shift_top1 = natural_shift_top1 + message[3]/len(extra_messages)
                    ooc_corr_top1 = ooc_corr_top1 + message[4]/len(extra_messages)
                    mixed_top1 = mixed_top1 + message[5]/len(extra_messages)
                    self.global_rep = self.global_rep + torch.tensor(message[6:6+self.conf.rep_len])/len(extra_messages)
                    #self.global_rep_sigma = self.global_rep_sigma + torch.tensor(message[6+self.conf.rep_len:6+2*self.conf.rep_len])/len(extra_messages)
                    
            self.perf["top1"] = top1.item()
            self.perf["corr_top1"] = corr_top1.item()
            self.perf["ooc_top1"] = ooc_top1.item()
            self.perf["natural_shift_top1"] = natural_shift_top1.item()
            self.perf["ooc_corr_top1"] = ooc_corr_top1.item()
            self.perf["mixed_top1"] = mixed_top1.item()
            self.perf["global_top1"] = global_top1_perfs
            fc= [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            
            for message in extra_messages.values():
                    fc[0] = (fc[0] + (message[0]-top1)**2).item()
                    fc[1] = (fc[1] + (message[1]-corr_top1)**2).item()
                    fc[2] = (fc[2] + (message[2]-ooc_top1)**2).item()
                    fc[3] = (fc[3]+ (message[3]-natural_shift_top1)**2).item()
                    fc[4] = (fc[4] + (message[4]-ooc_corr_top1)**2).item()
                    fc[5] = (fc[5] + (message[5]-mixed_top1)**2).item()
            # logging.
            display_perf(self.conf, self.perf)
            #display_perf(self.conf, fc)
    def _check_early_stopping(self):
        # to use early_stopping checker, we need to ensure patience > 0.
        meet_flag = False

        # consider both of target_perf and early_stopping
        if self.conf.target_perf is not None:
            assert 100 >= self.conf.target_perf > 0

            # meet the target perf.
            if (
                    self.coordinator.key_metric.cur_perf is not None
                    and self.coordinator.key_metric.cur_perf > self.conf.target_perf
            ):
                self.logger.log("Master early stopping: meet target perf.")
                self.meet_target = True
                meet_flag = True
            # or has no progress and early stop it.
            elif self.early_stopping_tracker(self.coordinator.key_metric.cur_perf):
                self.logger.log(
                    "Master early stopping: not meet target perf but has no patience."
                )
                meet_flag = True

        # only consider the early stopping.
        else:
            if self.early_stopping_tracker(self.coordinator.key_metric.cur_perf):
                meet_flag = True

        if meet_flag:
            # we perform the early-stopping check:
            # (1) before the local training and (2) after the update of the comm_round.
            _comm_round = self.comm_round - 1
            self.comm_round = -1
            self._finishing(_comm_round)

    def _finishing(self, _comm_round=None):
        self.logger.save_json()
        self.logger.log(f"Master finished the federated learning.")
        self.is_finished = True
        self.finished_comm = _comm_round
        checkpoint.save_arguments(self.conf)
        os.system(f"echo {self.conf.checkpoint_root} >> {self.conf.job_id}")
    

    def recieve_ddes(self,selected_client_ids):
        ###print("enter heads")
        self.dde_models=dict()
        for selected_client_id in selected_client_ids:
            #self.dde_models[selected_client_id]=MlpModule(n_input=64,n_hidden=16,n_layers=self.conf.dde_layers, last_layer_dim=1, activation=torch.nn.Softplus)
            self.dde_models[selected_client_id]=MlpModule(n_input=self.conf.rep_len,n_hidden=16,n_layers=self.conf.dde_layers, last_layer_dim=1, activation=torch.nn.Softplus)
        flatten_heads=dict()
        dist.barrier()
        for selected_client_id in selected_client_ids:
            client_tb_memory = TensorBuffer(
                list(self.dde_models[1].state_dict().values()))
            client_tb_memory.buffer = torch.zeros_like(client_tb_memory.buffer)
            flatten_heads[selected_client_id] = client_tb_memory

        reqs = {}
        for client_id, world_id in zip(selected_client_ids, self.world_ids):
            req = dist.irecv(tensor=flatten_heads[client_id].buffer,
                             src=world_id)
            reqs[client_id] = req

        for client_id, req in reqs.items():
            req.wait()
        dist.barrier()

        for selected_client_id in selected_client_ids:
            head_state_dict=self.dde_models[selected_client_id].state_dict()
            flatten_heads[selected_client_id].unpack(
                head_state_dict.values())
            self.dde_models[selected_client_id].load_state_dict(head_state_dict)

    def recieve_heads(self,selected_client_ids):
        ###print("enter heads")
        self.head_models=dict()
        for selected_client_id in selected_client_ids:
            self.head_models[selected_client_id]=torch.nn.Linear(self.conf.rep_len, utils.get_num_classes(self.conf.data), bias=False)
        flatten_heads=dict()
        dist.barrier()
        for selected_client_id in selected_client_ids:
            client_tb_memory = TensorBuffer(
                list(self.head_models[1].state_dict().values()))
            client_tb_memory.buffer = torch.zeros_like(client_tb_memory.buffer)
            flatten_heads[selected_client_id] = client_tb_memory

        reqs = {}
        for client_id, world_id in zip(selected_client_ids, self.world_ids):
            req = dist.irecv(tensor=flatten_heads[client_id].buffer,
                             src=world_id)
            reqs[client_id] = req

        for client_id, req in reqs.items():
            req.wait()
        dist.barrier()

        for selected_client_id in selected_client_ids:
            head_state_dict=self.head_models[selected_client_id].state_dict()
            flatten_heads[selected_client_id].unpack(
                head_state_dict.values())
            self.head_models[selected_client_id].load_state_dict(head_state_dict)
        ###print("leave heads")

    #接收某种测试数据，构建所有客户某一测试类型的 dataset(接收到的是model,构建的是数据集)
    def recieve_qs(self,selected_client_ids,name):
        ###print("enter a q")
        q_models=dict()
        for selected_client_id in selected_client_ids:
            q_models[selected_client_id]=Queue("cpu",self.conf.rep_len,500)
        flatten_qs=dict()
        dist.barrier()
        for selected_client_id in selected_client_ids:
            client_tb_memory = TensorBuffer(
                list(q_models[1].state_dict().values()))
            client_tb_memory.buffer = torch.zeros_like(client_tb_memory.buffer)
            flatten_qs[selected_client_id] = client_tb_memory

        reqs = {}
        for client_id, world_id in zip(selected_client_ids, self.world_ids):
            req = dist.irecv(tensor=flatten_qs[client_id].buffer,
                             src=world_id)
            reqs[client_id] = req

        for client_id, req in reqs.items():
            req.wait()
        dist.barrier()

        for selected_client_id in selected_client_ids:
            q_state_dict=q_models[selected_client_id].state_dict()
            flatten_qs[selected_client_id].unpack(
               q_state_dict.values())
            q_models[selected_client_id].load_state_dict(q_state_dict)
        
        for selected_client_id in selected_client_ids:
            self.test_feature_sets[selected_client_id][name] = q_models[selected_client_id].get_valid_dataset()
            #data_loader = DataLoader(self.test_feature_sets[selected_client_id][name], batch_size=1, shuffle=False)
            # 初始化一个空列表来存储所有样本
            all_samples = []
            for idx in range(len(self.test_feature_sets[selected_client_id][name])):
                all_samples.append(self.test_feature_sets[selected_client_id][name][idx][0])
            
            all_samples_tensor = torch.stack(all_samples)
            self.test_feature_matrix[selected_client_id][name]=all_samples_tensor
            ###print(len(self.test_feature_sets[selected_client_id][name]))
        ###print("leave a q")
    
    #接收各种测试数据，构建所有客户所有测试类型的 datasets
    def recive_all_qs(self,selected_client_ids):
        ###print("enter qs")
       
        self.recieve_qs(selected_client_ids, "test_loader")
       
        self.recieve_qs(selected_client_ids, "corr_test_loader")
        self.recieve_qs(selected_client_ids, "ooc_test_loader")
        self.recieve_qs(selected_client_ids, "natural_shift_test_loader")
        self.recieve_qs(selected_client_ids, "ooc_corr_test_loader")
        self.recieve_qs(selected_client_ids, "mixed_test_loader")
        ###print("leave qs")
    
    def get_acc_sim(self,dataset,client_id,selected_client_ids):
        trainloader = DataLoader(dataset, batch_size=500, shuffle=False)
        corrects=0
        #with torch.no_grad():
        for features, labels in trainloader:
                # 初始化最大值列表和最大索引列表##print
                print("client "+str(client_id)+":"+"（ l2 with dynamic)")
                #max_values = torch.full((features.size(0),1), -float('inf'))
                max_values = torch.full((features.size(0),1), float('inf'))
                indices = torch.zeros((features.size(0),1), dtype=torch.long)  # 初始化为0
                for index in selected_client_ids:
                    
                    reps=self.test_feature_matrix[index]["test_loader"]
                    
                    Q=features.unsqueeze(1).repeat(1,reps.size(0),1)
                    T=reps.unsqueeze(0).repeat(features.size(0),1,1)
                    distances = (Q - T).pow(2).sum(2).sqrt()
                    '''
                    if index==client_id:
                        i=1
                    else:
                        i=0
                    '''
                    outputs,_ = torch.topk(distances, k=self.conf.masterk, dim=1, largest=False, sorted=True)
                    #outputs_real=outputs[:,i].unsqueeze(-1)
                    #outputs_real=outputs.sum(dim=1, keepdim=True)
                    outputs_real=outputs[:,1:].sum(dim=1, keepdim=True)
                    #max_values = torch.max(max_values, outputs)  # 更新最大值
                    max_values = torch.min(max_values, outputs_real)
                    indices = torch.where(max_values == outputs_real, index, indices)
                print(indices.view(len(indices))[:30])
                print()
                   
            

                unique_indices, inverse_indices = torch.unique(indices, return_inverse=True)
                ###print(unique_indices)
                ###print(torch.where(indices == torch.Tensor([1]))[0])
                features_for_heads = {index.item():torch.gather(features, 0, torch.where(indices == index)[0].unsqueeze(1).repeat(1,self.conf.rep_len)) for index in unique_indices}
                labels_for_heads = {index.item():torch.gather(labels, 0, torch.where(indices == index)[0]) for index in unique_indices}
                #features_for_heads= features.index_select(0, torch.tensor(index))
                #features_for_heads = torch.split(features, index=index.tolist(), dim=0)
                #labels_for_heads = torch.split(labels , index=index.tolist())              
                for i in  unique_indices:
                    correct=0
                    model=self.head_models[i.item()]
                    features=features_for_heads[i.item()]
                    labels=labels_for_heads[i.item()]
                    output=model(features)
                    predicted = torch.argmax(output, dim=1)
                    correct = (predicted == labels).sum().item()
                    corrects+=correct
        return 100*corrects/len(dataset)
    #dde,abondon
    def get_acc(self,dataset,client_id):
        trainloader = DataLoader(dataset, batch_size=500, shuffle=False)
        corrects=0
        #with torch.no_grad():
        for features, labels in trainloader:
                # 初始化最大值列表和最大索引列表
                ##print("client "+str(client_id)+":")
                max_values = torch.full((features.size(0),1), -float('inf'))
                #max_values = torch.full((features.size(0),1), float('inf'))
                indices = torch.zeros((features.size(0),1), dtype=torch.long)  # 初始化为0
                for index,model in self.dde_models.items():
                    outputs=model(features,True)
                    '''
                    features.requires_grad_(True)
                    prob=model(features,True)
                    grad = torch.autograd.grad(prob, features, grad_outputs=torch.ones(prob.shape, device=prob.device), create_graph=True)[0]
                    outputs=torch.norm(grad,p=2)
                    '''
                    max_values = torch.max(max_values, outputs)  # 更新最大值
                    #max_values = torch.min(max_values, outputs)
                    indices = torch.where(max_values == outputs, index, indices)
                ##print(indices.view(len(indices))[:30])
                #exit()

                unique_indices, inverse_indices = torch.unique(indices, return_inverse=True)
                ###print(unique_indices)
                ###print(torch.where(indices == torch.Tensor([1]))[0])
                features_for_heads = {index.item():torch.gather(features, 0, torch.where(indices == index)[0].unsqueeze(1).repeat(1,self.conf.rep_len)) for index in unique_indices}
                labels_for_heads = {index.item():torch.gather(labels, 0, torch.where(indices == index)[0]) for index in unique_indices}
                #features_for_heads= features.index_select(0, torch.tensor(index))
                #features_for_heads = torch.split(features, index=index.tolist(), dim=0)
                #labels_for_heads = torch.split(labels , index=index.tolist())              
                for i in  unique_indices:
                    correct=0
                    model=self.head_models[i.item()]
                    features=features_for_heads[i.item()]
                    labels=labels_for_heads[i.item()]
                    output=model(features)
                    predicted = torch.argmax(output, dim=1)
                    correct = (predicted == labels).sum().item()
                    corrects+=correct
        return 100*corrects/len(dataset)
    
    def get_acc_by_oracle(self,dataset,client_id):
        corrects=0
        trainloader = DataLoader(dataset, batch_size=500, shuffle=False)
        model=self.head_models[client_id]
       
        for features, labels in trainloader:
            output=model(features)
            predicted = torch.argmax(output, dim=1)
            correct = (predicted == labels).sum().item()
            corrects+=correct
        return 100*corrects/len(dataset)
    
    def get_real_extra_message_for_ft_rep(self,selected_client_ids,extra_messages):
        self.test_feature_sets=dict()
        self.test_feature_matrix=dict()
        self.reps=torch.rand(self.conf.n_clients,self.conf.rep_len,requires_grad=False)
        self.ccreps=torch.rand(self.conf.n_clients,self.conf.rep_len,requires_grad=False)
        for client_id in selected_client_ids:
            self.test_feature_sets[client_id]=dict()
            self.test_feature_matrix[client_id]=dict()
            self.reps[client_id-1] = extra_messages[client_id][6:6+self.conf.rep_len]
            self.ccreps[client_id-1]=self.cc_layer.weight.data[client_id-1]
        self.recieve_heads(selected_client_ids)
        #self.aggregation_head(list(self.client_sampler.selected_client_ids))
        #self.recieve_ddes(selected_client_ids)
        self.recieve_pros(selected_client_ids)
        self.recive_all_qs(selected_client_ids)
        '''
        whole_data_matrix=torch.cat([self.test_feature_matrix[selected_client_id]["test_loader"] for selected_client_id in selected_client_ids],dim=0)
        
        for client_id in selected_client_ids:
            AveLogP=AverageMeter()
            dde_real_logP_real =  self.dde_models[client_id](whole_data_matrix)
            mean_logP= dde_real_logP_real.mean(dim=0).item()
            AveLogP.update(mean_logP,len(whole_data_matrix))
            self.dde_models[client_id].average.data=torch.Tensor([AveLogP.avg])
            ##print("client "+str(client_id)+":")
            ##print(AveLogP.avg)
            ##print(AveLogP.sum)
            ##print(AveLogP.count)
            ##print()
        '''
        with torch.no_grad():
            if self.comm_round%10 ==0: 
                for client_id in selected_client_ids: 
                
                    extra_messages[client_id][0]=self.get_acc_by_oracle(self.test_feature_sets[client_id]["test_loader"],client_id)
                    extra_messages[client_id][1]=self.get_acc_by_rep_dis(self.test_feature_sets[client_id]["test_loader"],client_id)
                    extra_messages[client_id][2]=self.get_acc_by_rep_dis_cc(self.test_feature_sets[client_id]["test_loader"],client_id)
                    extra_messages[client_id][3]=self.get_acc_by_pro(self.test_feature_sets[client_id]["test_loader"],client_id)
                    extra_messages[client_id][4]=self.get_acc_sim(self.test_feature_sets[client_id]["test_loader"],client_id,selected_client_ids)
                    extra_messages[client_id][5]=self.get_acc_sim_l1(self.test_feature_sets[client_id]["test_loader"],client_id,selected_client_ids)
                    '''
                    extra_messages[client_id][1]=self.get_acc(self.test_feature_sets[client_id]["corr_test_loader"],client_id)
                    extra_messages[client_id][2]=self.get_acc(self.test_feature_sets[client_id]["ooc_test_loader"],client_id)
                    extra_messages[client_id][3]=self.get_acc(self.test_feature_sets[client_id]["natural_shift_test_loader"],client_id)
                    extra_messages[client_id][4]=self.get_acc(self.test_feature_sets[client_id]["ooc_corr_test_loader"],client_id)
                    extra_messages[client_id][5]=self.get_acc(self.test_feature_sets[client_id]["mixed_test_loader"],client_id)
                    '''
        return extra_messages
    
    def aggregation_dde(self, index):
       
        #self.dde=MlpModule(n_input=64,n_hidden=16,n_layers=self.conf.dde_layers, last_layer_dim=1, activation=torch.nn.Softplus)
        self.dde=MlpModule(n_input=self.conf.rep_len,n_hidden=16,n_layers=self.conf.dde_layers, last_layer_dim=1, activation=torch.nn.Softplus)
        params = {}
        for k, v in self.dde_models[1].named_parameters():
            params[k] = torch.zeros_like(v.data)
        for j in index:
            for k, v in self.dde_models[j].named_parameters():
                params[k] += v.data / len(index)
        for k, v in self.dde.named_parameters():
            v.data = params[k].data.clone()
    
    def aggregation_pro(self, index):
       
        #self.dde=MlpModule(n_input=64,n_hidden=16,n_layers=self.conf.dde_layers, last_layer_dim=1, activation=torch.nn.Softplus)
        self.global_pro=CategoryDistributionModel(self.conf.rep_len,self.conf.q_level)
        params = {}
        for k, v in self.pro_models[1].named_parameters():
            params[k] = torch.zeros_like(v.data)
        for j in index:
            for k, v in self.pro_models[j].named_parameters():
                params[k] += v.data / len(index)
        for k, v in self.global_pro.named_parameters():
            v.data = params[k].data.clone()
    
    def aggregation_head(self, index):
       
        #self.dde=MlpModule(n_input=64,n_hidden=16,n_layers=self.conf.dde_layers, last_layer_dim=1, activation=torch.nn.Softplus)
        self.trial= nn.Linear(self.conf.rep_len, self.conf.n_clients, bias=False)
        params = {}
        for k, v in self.head_models[1].named_parameters():
            params[k] = torch.zeros_like(v.data)
        for j in index:
            for k, v in self.head_models[j].named_parameters():
                params[k] += v.data / len(index)
        for k, v in self.trial.named_parameters():
            v.data = params[k].data.clone()

    def recieve_pros(self,selected_client_ids):
        ###print("enter heads")
        self.pro_models=dict()
        for selected_client_id in selected_client_ids:
            #self.dde_models[selected_client_id]=MlpModule(n_input=64,n_hidden=16,n_layers=self.conf.dde_layers, last_layer_dim=1, activation=torch.nn.Softplus)
            self.pro_models[selected_client_id]=CategoryDistributionModel(self.conf.rep_len,self.conf.q_level)
        flatten_heads=dict()
        dist.barrier()
        for selected_client_id in selected_client_ids:
            client_tb_memory = TensorBuffer(
                list(self.pro_models[1].state_dict().values()))
            client_tb_memory.buffer = torch.zeros_like(client_tb_memory.buffer)
            flatten_heads[selected_client_id] = client_tb_memory

        reqs = {}
        for client_id, world_id in zip(selected_client_ids, self.world_ids):
            req = dist.irecv(tensor=flatten_heads[client_id].buffer,
                             src=world_id)
            reqs[client_id] = req

        for client_id, req in reqs.items():
            req.wait()
        dist.barrier()

        for selected_client_id in selected_client_ids:
            head_state_dict=self.pro_models[selected_client_id].state_dict()
            flatten_heads[selected_client_id].unpack(
                head_state_dict.values())
            self.pro_models[selected_client_id].load_state_dict(head_state_dict)


    def _initial_dde_to_selected_clients_with_uniform_noise(self,selected_client_ids):
        #self.dde= MlpModule(n_input=64,n_hidden=16,n_layers=self.conf.dde_layers, last_layer_dim=1, activation=nn.Softplus).to(self.graph.device)
        self.dde= MlpModule(n_input=self.conf.rep_len,n_hidden=16,n_layers=self.conf.dde_layers, last_layer_dim=1, activation=nn.Softplus).to(self.graph.device)
        self.dde_optm = torch.optim.Adam(self.dde.parameters(), lr=0.001)
        uniform_data = torch.rand(20000, self.conf.rep_len) * 2 - 1
        uniform_data =uniform_data.to(self.graph.device)
        uniform_dataset = UniformDataset(uniform_data)
        uniform_data_loader = torch.utils.data.DataLoader(uniform_dataset, batch_size=512, shuffle=True)
        '''
        for e in range(self.conf.dde_epoch):
            for batch_data in uniform_data_loader :
                self.dde_optm.zero_grad()
                noise_rep=_get_noisy(batch_data,self.conf.sigma)
                dde_real_output_real, dde_real_logP_real = _get_dde_output(self.conf.sigma, self.dde, noise_rep)
                loss_dde_real = F.mse_loss(dde_real_output_real,batch_data, reduction='sum') / 2#+0.00001*dde_real_logP_real.pow(2).mean(dim=0)
                loss_dde_real.backward()
                self.dde_optm.step()
        '''
    def _send_pro(self,selected_client_ids):
        self.global_pro=self.global_pro.cpu()
        for worker_rank, selected_client_id in enumerate(selected_client_ids, start=1):
            client_model_state_dict =  self.global_pro.state_dict()
            flatten_model = TensorBuffer(list(client_model_state_dict.values()))
            dist.send(tensor=flatten_model.buffer, dst=worker_rank)
        dist.barrier()
    
    def _send_pros(self,selected_client_ids):
        for i in range(1,21):
            for worker_rank, selected_client_id in enumerate(selected_client_ids, start=1):
                client_model_state_dict =  self.pro_models[i].state_dict()
                flatten_model = TensorBuffer(list(client_model_state_dict.values()))
                dist.send(tensor=flatten_model.buffer, dst=worker_rank)
            dist.barrier()
    
    def _send_dde(self,selected_client_ids):
        self.dde=self.dde.cpu()
        for worker_rank, selected_client_id in enumerate(selected_client_ids, start=1):
            client_model_state_dict =  self.dde.state_dict()
            flatten_model = TensorBuffer(list(client_model_state_dict.values()))
            dist.send(tensor=flatten_model.buffer, dst=worker_rank)
        dist.barrier()
    
    def _send_head(self,selected_client_ids):
        self.trial=self.trial.cpu()
        for worker_rank, selected_client_id in enumerate(selected_client_ids, start=1):
            client_model_state_dict =  self.trial.state_dict()
            flatten_model = TensorBuffer(list(client_model_state_dict.values()))
            dist.send(tensor=flatten_model.buffer, dst=worker_rank)
        dist.barrier()
    def _send_cc_layer(self,selected_client_ids):
        self.cc_layer =self.cc_layer.cpu()
        for worker_rank, selected_client_id in enumerate(selected_client_ids, start=1):
            client_model_state_dict =  self.cc_layer.state_dict()
            flatten_model = TensorBuffer(list(client_model_state_dict.values()))
            dist.send(tensor=flatten_model.buffer, dst=worker_rank)
        dist.barrier()

    def get_acc_by_rep_dis(self,dataset,client_id):
        trainloader = DataLoader(dataset, batch_size=500, shuffle=False)
        corrects=0
        #with torch.no_grad():
        for features, labels in trainloader:
                print("client "+str(client_id)+":( l1 with fixed )")
                Q=fsq_quantize(features.unsqueeze(1).repeat(1,self.reps.size(0),1),self.conf.q_level)
                T=fsq_quantize(self.reps.unsqueeze(0).repeat(features.size(0),1,1),self.conf.q_level)
                indices = abs(Q - T).sum(2).min(1)[1]+1
                '''
                Q=features.unsqueeze(1).repeat(1,self.reps.size(0),1)
                T=self.reps.unsqueeze(0).repeat(features.size(0),1,1)
                indices=(Q-T).pow(2).sum(2).sqrt().min(1)[1]+1
                '''
                print(indices.view(len(indices))[:30])
                unique_indices, inverse_indices = torch.unique(indices, return_inverse=True)
                ###print(unique_indices)
                ###print(torch.where(indices == torch.Tensor([1]))[0])
                features_for_heads = {index.item():torch.gather(features, 0, torch.where(indices == index)[0].unsqueeze(1).repeat(1,self.conf.rep_len)) for index in unique_indices}
                labels_for_heads = {index.item():torch.gather(labels, 0, torch.where(indices == index)[0]) for index in unique_indices}
                #features_for_heads= features.index_select(0, torch.tensor(index))
                #features_for_heads = torch.split(features, index=index.tolist(), dim=0)
                #labels_for_heads = torch.split(labels , index=index.tolist())              
                for i in  unique_indices:
                    correct=0
                    model=self.head_models[i.item()]
                    features=features_for_heads[i.item()]
                    labels=labels_for_heads[i.item()]
                    output=model(features)
                    predicted = torch.argmax(output, dim=1)
                    correct = (predicted == labels).sum().item()
                    corrects+=correct
        return 100*corrects/len(dataset)
    
    def get_acc_by_rep_dis_cc(self,dataset,client_id):
        trainloader = DataLoader(dataset, batch_size=500, shuffle=False)
        corrects=0
        #with torch.no_grad():
        for features, labels in trainloader:
                print("client "+str(client_id)+":( l1 with fixed and predefined)")
                Q=fsq_quantize(features.unsqueeze(1).repeat(1,self.reps.size(0),1),self.conf.q_level)
                T=self.ccreps.unsqueeze(0).repeat(features.size(0),1,1)
                indices = abs(Q - T).sum(2).min(1)[1]+1
                '''
                Q=features.unsqueeze(1).repeat(1,self.reps.size(0),1)
                T=self.reps.unsqueeze(0).repeat(features.size(0),1,1)
                indices=(Q-T).pow(2).sum(2).sqrt().min(1)[1]+1
                '''
                print(indices.view(len(indices))[:30])
                unique_indices, inverse_indices = torch.unique(indices, return_inverse=True)
                ###print(unique_indices)
                ###print(torch.where(indices == torch.Tensor([1]))[0])
                features_for_heads = {index.item():torch.gather(features, 0, torch.where(indices == index)[0].unsqueeze(1).repeat(1,self.conf.rep_len)) for index in unique_indices}
                labels_for_heads = {index.item():torch.gather(labels, 0, torch.where(indices == index)[0]) for index in unique_indices}
                #features_for_heads= features.index_select(0, torch.tensor(index))
                #features_for_heads = torch.split(features, index=index.tolist(), dim=0)
                #labels_for_heads = torch.split(labels , index=index.tolist())              
                for i in  unique_indices:
                    correct=0
                    model=self.head_models[i.item()]
                    features=features_for_heads[i.item()]
                    labels=labels_for_heads[i.item()]
                    output=model(features)
                    predicted = torch.argmax(output, dim=1)
                    correct = (predicted == labels).sum().item()
                    corrects+=correct
        return 100*corrects/len(dataset)
    #abondon
    def get_acc_by_rep_dis_(self,dataset,client_id):
        trainloader = DataLoader(dataset, batch_size=500, shuffle=False)
        corrects=0
        #with torch.no_grad():
        for features, labels in trainloader:
                ##print("client "+str(client_id)+":(rep_dis_)")
                Q=features.unsqueeze(1).repeat(1,self.ccreps.size(0),1)
                T=self.ccreps.unsqueeze(0).repeat(features.size(0),1,1)
                indices=(Q-T).pow(2).sum(2).sqrt().min(1)[1]+1
                #otherindices=(Q-T).pow(2).sum(2).sqrt().max(1)[1]+1
                ##print(indices.view(len(indices))[:30])
                ###print(otherindices.view(len(indices))[:30])
                unique_indices, inverse_indices = torch.unique(indices, return_inverse=True)
                ###print(unique_indices)
                ###print(torch.where(indices == torch.Tensor([1]))[0])
                features_for_heads = {index.item():torch.gather(features, 0, torch.where(indices == index)[0].unsqueeze(1).repeat(1,self.conf.rep_len)) for index in unique_indices}
                labels_for_heads = {index.item():torch.gather(labels, 0, torch.where(indices == index)[0]) for index in unique_indices}
                #features_for_heads= features.index_select(0, torch.tensor(index))
                #features_for_heads = torch.split(features, index=index.tolist(), dim=0)
                #labels_for_heads = torch.split(labels , index=index.tolist())              
                for i in  unique_indices:
                    correct=0
                    model=self.head_models[i.item()]
                    features=features_for_heads[i.item()]
                    labels=labels_for_heads[i.item()]
                    output=model(features)
                    predicted = torch.argmax(output, dim=1)
                    correct = (predicted == labels).sum().item()
                    corrects+=correct
        return 100*corrects/len(dataset)
           

    def get_acc_by_ent(self,dataset,client_id):
        trainloader = DataLoader(dataset, batch_size=500, shuffle=False)
        corrects=0
       
        for features, labels in trainloader:
                max_values = torch.full((features.size(0),1), float('inf'))            
                indices = torch.zeros((features.size(0),1), dtype=torch.long)  # 初始化为0
                for index,model in self.head_models.items():
                    output=model(features)
                    probs = F.softmax(output, dim=1)
                    outputs= -(probs * (probs.log())).sum(dim=1).view(-1,1)
                    max_values = torch.min(max_values, outputs)  # 更新最大值               
                    indices = torch.where(max_values == outputs, index, indices)
                unique_indices, inverse_indices = torch.unique(indices, return_inverse=True)
                ###print(unique_indices)
                ###print(torch.where(indices == torch.Tensor([1]))[0])
                features_for_heads = {index.item():torch.gather(features, 0, torch.where(indices == index)[0].unsqueeze(1).repeat(1,self.conf.rep_len)) for index in unique_indices}
                labels_for_heads = {index.item():torch.gather(labels, 0, torch.where(indices == index)[0]) for index in unique_indices}
                #features_for_heads= features.index_select(0, torch.tensor(index))
                #features_for_heads = torch.split(features, index=index.tolist(), dim=0)
                #labels_for_heads = torch.split(labels , index=index.tolist())              
                for i in  unique_indices:
                    correct=0
                    model=self.head_models[i.item()]
                    features=features_for_heads[i.item()]
                    labels=labels_for_heads[i.item()]
                    output=model(features)
                    predicted = torch.argmax(output, dim=1)
                    correct = (predicted == labels).sum().item()
                    corrects+=correct
        return 100*corrects/len(dataset)
    
        
    def get_acc_by_rep_dis_mean_rak(self,dataset,client_id):
        trainloader = DataLoader(dataset, batch_size=500, shuffle=False)
        corrects=0
        #with torch.no_grad():
        for features, labels in trainloader:
                
                Q=features.unsqueeze(1).repeat(1,self.reps.size(0),1)
                T=self.reps.unsqueeze(0).repeat(features.size(0),1,1)
                top_values, top_indices = torch.topk((Q-T).pow(2).sum(2).sqrt(), k=10, largest=True, sorted=True)
                indices = top_indices[9] + 1
                unique_indices, inverse_indices = torch.unique(indices, return_inverse=True)
                ###print(unique_indices)
                ###print(torch.where(indices == torch.Tensor([1]))[0])
                features_for_heads = {index.item():torch.gather(features, 0, torch.where(indices == index)[0].unsqueeze(1).repeat(1,self.conf.rep_len)) for index in unique_indices}
                labels_for_heads = {index.item():torch.gather(labels, 0, torch.where(indices == index)[0]) for index in unique_indices}
                #features_for_heads= features.index_select(0, torch.tensor(index))
                #features_for_heads = torch.split(features, index=index.tolist(), dim=0)
                #labels_for_heads = torch.split(labels , index=index.tolist())              
                for i in  unique_indices:
                    correct=0
                    model=self.head_models[i.item()]
                    features=features_for_heads[i.item()]
                    labels=labels_for_heads[i.item()]
                    output=model(features)
                    predicted = torch.argmax(output, dim=1)
                    correct = (predicted == labels).sum().item()
                    corrects+=correct
        return 100*corrects/len(dataset)
    
    def get_acc_sim_l1(self,dataset,client_id,selected_client_ids):
        trainloader = DataLoader(dataset, batch_size=500, shuffle=False)
        corrects=0
        #with torch.no_grad():
        for features, labels in trainloader:
                # 初始化最大值列表和最大索引列表##print
                
                #max_values = torch.full((features.size(0),1), -float('inf'))
                max_values = torch.full((features.size(0),1), float('inf'))
                indices = torch.zeros((features.size(0),1), dtype=torch.long)  # 初始化为0
                print("client "+str(client_id)+":"+"( l1 with dynamic)")
                for index in selected_client_ids:
                   
                    reps=self.test_feature_matrix[index]["test_loader"]
                    
                    Q=fsq_quantize(features.unsqueeze(1).repeat(1,reps.size(0),1),self.conf.q_level)
                    T=fsq_quantize(reps.unsqueeze(0).repeat(features.size(0),1,1),self.conf.q_level)
                    distances = abs(Q - T).sum(2)
                    '''
                    features=fsq_quantize(features,self.conf.q_level)/self.conf.q_level
                    reps=fsq_quantize(reps,self.conf.q_level)/self.conf.q_level
                    distances = -torch.mm(F.normalize(features), F.normalize(reps).transpose(1, 0))  
                    '''
                    '''
                    if index==client_id:
                        i=1
                    else:
                        i=0
                    '''
                    outputs,_ = torch.topk(distances, k=self.conf.masterk, dim=1, largest=False, sorted=True)
                    #outputs_real=outputs[:,i].unsqueeze(-1)
                    #outputs_real=outputs.sum(dim=1, keepdim=True)
                    outputs_real=outputs[:,1:].sum(dim=1, keepdim=True)
                    #max_values = torch.max(max_values, outputs)  # 更新最大值
                    max_values = torch.min(max_values, outputs_real)
                    indices = torch.where(max_values == outputs_real, index, indices)
                print(indices.view(len(indices))[:30])
                print(max_values.view(len(max_values))[:30])
                print()
                
            

                unique_indices, inverse_indices = torch.unique(indices, return_inverse=True)
                
                ###print(unique_indices)
                ###print(torch.where(indices == torch.Tensor([1]))[0])
                features_for_heads = {index.item():torch.gather(features, 0, torch.where(indices == index)[0].unsqueeze(1).repeat(1,self.conf.rep_len)) for index in unique_indices}
                labels_for_heads = {index.item():torch.gather(labels, 0, torch.where(indices == index)[0]) for index in unique_indices}
                #features_for_heads= features.index_select(0, torch.tensor(index))
                #features_for_heads = torch.split(features, index=index.tolist(), dim=0)
                #labels_for_heads = torch.split(labels , index=index.tolist())              
                for i in  unique_indices:
                    correct=0
                    model=self.head_models[i.item()]
                    features=features_for_heads[i.item()]
                    labels=labels_for_heads[i.item()]
                    output=model(features)
                    predicted = torch.argmax(output, dim=1)
                    correct = (predicted == labels).sum().item()
                    corrects+=correct
        return 100*corrects/len(dataset)
    
    def get_acc_by_pro(self,dataset,client_id):
        trainloader = DataLoader(dataset, batch_size=500, shuffle=False)
        corrects=0
        #with torch.no_grad():
        for features, labels in trainloader:
                # 初始化最大值列表和最大索引列表
                print("client "+str(client_id)+":"+"(pro)")
                features_temp=fsq_quantize(features,self.conf.q_level).to(torch.int64)
               
                max_values = torch.full((features_temp.size(0),1), -float('inf'))
                #max_values = torch.full((features.size(0),1), float('inf'))
                indices = torch.zeros((features_temp.size(0),1), dtype=torch.long)  # 初始化为0
                for index,model in self.pro_models.items():
                   
                    outputs=model(features_temp)
                    
                    max_values = torch.max(max_values, outputs)  # 更新最大值
                    
                    indices = torch.where(max_values == outputs, index, indices)
                ##print(indices.shape)
                print(indices.view(len(indices))[:30])
                print()

                unique_indices, inverse_indices = torch.unique(indices, return_inverse=True)
                
                features_for_heads = {index.item():torch.gather(features, 0, torch.where(indices == index)[0].unsqueeze(1).repeat(1,self.conf.rep_len)) for index in unique_indices}
                labels_for_heads = {index.item():torch.gather(labels, 0, torch.where(indices == index)[0]) for index in unique_indices}
                          
                for i in  unique_indices:
                    correct=0
                    model=self.head_models[i.item()]
                    features=features_for_heads[i.item()]
                    labels=labels_for_heads[i.item()]
                    output=model(features)
                    predicted = torch.argmax(output, dim=1)
                    correct = (predicted == labels).sum().item()
                    corrects+=correct
        return 100*corrects/len(dataset)
    
    def get_acc_by_pro_log(self,dataset,client_id):
        trainloader = DataLoader(dataset, batch_size=500, shuffle=False)
        corrects=0
        #with torch.no_grad():
        for features, labels in trainloader:
                # 初始化最大值列表和最大索引列表
                print("client "+str(client_id)+":"+"(log)")
                features_temp=fsq_quantize(features,self.conf.q_level).to(torch.int64)
               
                max_values = torch.full((features_temp.size(0),1), -float('inf'))
                #max_values = torch.full((features.size(0),1), float('inf'))
                indices = torch.zeros((features_temp.size(0),1), dtype=torch.long)  # 初始化为0
                for index,model in self.pro_models.items():
                   
                    outputs=model(features_temp,log=True)
                    
                    max_values = torch.max(max_values, outputs)  # 更新最大值
                    
                    indices = torch.where(max_values == outputs, index, indices)
                ##print(indices.shape)
                print(indices.view(len(indices))[:30])
                print()

                unique_indices, inverse_indices = torch.unique(indices, return_inverse=True)
                
                features_for_heads = {index.item():torch.gather(features, 0, torch.where(indices == index)[0].unsqueeze(1).repeat(1,self.conf.rep_len)) for index in unique_indices}
                labels_for_heads = {index.item():torch.gather(labels, 0, torch.where(indices == index)[0]) for index in unique_indices}
                          
                for i in  unique_indices:
                    correct=0
                    model=self.head_models[i.item()]
                    features=features_for_heads[i.item()]
                    labels=labels_for_heads[i.item()]
                    output=model(features)
                    predicted = torch.argmax(output, dim=1)
                    correct = (predicted == labels).sum().item()
                    corrects+=correct
        return 100*corrects/len(dataset)


           
               
              
              
              
            
        
              
              
            
        
def fsq_quantize(tensor, num_values):
    #sigmoid_output = torch.sigmoid(tensor)
    tanh_output =torch.tanh(tensor)
    normalized = tanh_output * (num_values - 1)
    quantized_output = torch.round(normalized)
    return quantized_output


def l1(A, B):
    abs_diff_A = torch.abs(A.unsqueeze(2) - B.unsqueeze(0))

    l1_distances = torch.sum(abs_diff_A, dim=1)

    return l1_distances


def create_balanced_int_matrix(rows, cols, k):  
    # 检查k是否小于或等于cols  
    if k > cols:  
        raise ValueError(f"k ({k}) must be less than or equal to cols ({cols}).")  
      
    # 初始化一个全为0的矩阵  
    binary_matrix = torch.zeros(rows, cols, dtype=torch.int)  
      
    # 为每行随机分配数字  
    for i in range(rows):  
        # 生成当前行的随机索引  
        row_indices = torch.randperm(cols)  
          
        # 计算每个数字应该出现的次数  
        occurrences = cols // k  
          
        # 计算剩余的空间  
        remainder = cols % k  
          
        # 分配数字  
        for j in range(k):  
            # 计算当前数字应该填充的起始和结束索引  
            start = j * occurrences  
            if j < remainder:  
                end = start + occurrences + 1  
            else:  
                end = start + occurrences  
              
            # 在对应索引位置填充当前数字  
            binary_matrix[i, row_indices[start:end]] = j  
      
    return binary_matrix  
