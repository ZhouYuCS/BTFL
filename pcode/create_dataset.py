# -*- coding: utf-8 -*-
import functools
import torch
from torch.utils.data import ConcatDataset, Subset
from pcode.datasets.partition_data import DataPartitioner, ConcatPartition
from pcode.datasets.partition_data import record_class_distribution, partition_by_other_histogram
import pcode.datasets.prepare_data as prepare_data
import pcode.datasets.corr_data as corr_data
import torchvision.transforms as transforms
import numpy as np
from torch.distributions.dirichlet import Dirichlet
from torch.distributions.poisson import Poisson
import random
"""create dataset and load the data_batch."""


def load_data_batch(conf, _input, _target, is_training=True):
    """Load a mini-batch and record the loading time."""
    # do the transform right before using the data.
    transform = _get_transform(conf.data, is_training=is_training)
    if transform is not None:
        _input = transform(_input)
    if conf.graph.on_cuda:
        _input, _target = _input.cuda(), _target.cuda()
    _data_batch = {"input": _input, "target": _target}
    return _data_batch


def _get_transform(data_name, is_training):
    if data_name == "cifar10":
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
        if is_training:
            transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop((32, 32), 4),
                    normalize,
                ])
        else:
            transform = transforms.Compose([normalize])

    elif data_name == "cifar100":
        normalize = transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        if is_training:
            transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop((32, 32), 4),
                    normalize,
                ])
        else:
            transform = transforms.Compose([normalize])

    elif "imagenet" in data_name:
        normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        # normalize = transforms.Normalize((0.4810, 0.4574, 0.4078), (0.2146, 0.2104, 0.2138))
        if is_training:
            transform = transforms.Compose(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    normalize,
                ]
            )
        else:
            transform = transforms.Compose([normalize])

    elif data_name == "cinic":
        # decide normalize parameter
        normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

        # decide data type.
        if is_training:
            transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop((32, 32), 4),
                    normalize,
                ]
            )
        else:
            transform = transforms.Compose([normalize])

    elif data_name == "svhn":
        normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        transform = transforms.Compose([normalize])
    elif data_name == "femnist":
        transform = None
    elif data_name=="oh":
        transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    elif "pseudo_imagenet" in data_name:
        raise NotImplementedError
    elif "stl10" in data_name:
        raise NotImplementedError
    else:
        raise NotImplementedError

    return transform


class FLData:
    def __init__(
        self,
        conf,
        logger,
        graph,
        random_state,
        batch_size,
        img_resolution,
        corr_seed,
        corr_severity,
        local_n_epochs,
        num_workers,
        pin_memory,
    ):
        self.conf = conf
        self.logger = logger
        self.graph = graph
        self.random_state = random_state

        self.batch_size = batch_size
        self.img_resolution = img_resolution
        self.corr_seed = corr_seed
        self.corr_severity = corr_severity
        self.local_n_epochs = local_n_epochs
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def define_dataset(
        self,
        data_name,
        data_dir,
        is_personalized,
        test_partition_ratio,
        display_log=True,
        extra_arg=None,
    ):
        """This function manipulates the original dataset (before partitioning by clients):
            1. the train_dataset will be splitted into train_dataset and val_dataset.
            2. part of test_dataset will be combined with the train_dataset.
        """
        # prepare general train/test, and create the validation from train if necessary.
        train_dataset, val_dataset, test_dataset = self._define_tr_val_te_dataset(
            train_dataset=prepare_data.get_dataset(
                data_name, data_dir, split="train", img_resolution=self.img_resolution
            ),
            test_dataset=prepare_data.get_dataset(
                data_name, data_dir, split="test", img_resolution=self.img_resolution
            ),
             val_data_ratio=self.conf.val_data_ratio
        )

        # prepare the natural shifted class
        if data_name == "cifar10" :
            self.data_name = "cifar10"
            assert extra_arg == "cifar10.1"
            natural_shift_test_ds = prepare_data.get_dataset(
                data_name + ".1", data_dir, split="test", img_resolution=self.img_resolution
            )
        elif "imagenet" in data_name:
            self.data_name = "imagenet"
            self.natural_shift_list =["imagenet_r"]# ["imagenet_r", "imagenet_v2_topimages"] 
            natural_shift_test_ds = prepare_data.get_dataset(
                    "imagenet_r", data_dir, split="test", img_resolution=self.img_resolution
                )
        else:
            self.data_name="oh"
            natural_shift_test_ds = prepare_data.get_dataset(
                    "oh_n", data_dir, split="test", img_resolution=self.img_resolution
                )

        # merge train set and part of test set, keep the remaining test set as backup.
        if is_personalized:
            train_dataset, test_dataset = self._define_dataset_for_personalization(
                train_dataset, test_dataset, test_partition_ratio, display_log
            )

        if display_log:
            self.logger.log(
                "Data stat for original dataset: we have {} samples for train, {} samples for val, {} samples for test.".format(
                    len(train_dataset),
                    len(val_dataset) if val_dataset is not None else 0,
                    len(test_dataset),
                )
            )
        self.dataset = {
            "train": train_dataset,
            "val": val_dataset,
            "test": test_dataset,
            "natural_shift_test": natural_shift_test_ds,
        }

    def _define_tr_val_te_dataset(
        self, train_dataset, test_dataset, train_data_ratio=1.0, val_data_ratio=0.0
    ):
        assert val_data_ratio >= 0
        partition_sizes = [
            (1 - val_data_ratio) * train_data_ratio,
            (1 - val_data_ratio) * (1 - train_data_ratio),
            val_data_ratio,
        ]

        data_partitioner = DataPartitioner(
            train_dataset,
            partition_sizes,
            partition_type="random",
            partition_alphas=None,
            consistent_indices=False,
            random_state=self.random_state,
            graph=self.graph,
            logger=self.logger,
        )
        train_dataset = data_partitioner.use(0)

        # split for val data.
        if val_data_ratio > 0:
            val_dataset = data_partitioner.use(2)
            test_dataset=data_partitioner.use(1)
            return train_dataset, val_dataset, test_dataset
        else:
            return train_dataset, None, test_dataset

    def _define_dataset_for_personalization(
        self, train_dataset, test_dataset, test_partition_ratio, display_log
    ):
        """Create datasets for personalized federated learning scenarios, 
            based on train_dataset and test_dataset.
        """
        assert test_partition_ratio >= 0

        # split testset and move some of the testsets to the trainset.
        partition_sizes = [test_partition_ratio, 1 - test_partition_ratio]
        data_partitioner = DataPartitioner(
            test_dataset,
            partition_sizes,
            partition_type="random",
            partition_alphas=None,
            consistent_indices=False,
            random_state=self.random_state,
            graph=self.graph,
            logger=self.logger,
        )

        # partitioned test set
        if test_partition_ratio > 0:
            # merge train set and part of test set, keep the remaining test set as backup
            train_dataset = ConcatPartition([train_dataset, data_partitioner.use(0)])
            test_dataset = data_partitioner.use(1)

            if display_log:
                self.logger.log(
                    "Personalized setting: merge train set and part of test set."
                )
            return train_dataset, test_dataset
        else:
            # Note: test_dataset is a Dataset object, not a Partition object.
            # return None, test_dataset
            # Note: this will return a Partition object
            return train_dataset, test_dataset

    def inter_clients_data_partition(self, dataset, n_clients, partition_data_conf):
        if "size_conf" not in partition_data_conf:
            partition_sizes = [1.0 / n_clients for _ in range(n_clients)]
        else:
            # a few simple cases.
            # size_conf=1:2:3 indicates three data fraction cases.
            cases = [float(x) for x in partition_data_conf["size_conf"].split(":")]
            cases = (
                (cases * int(n_clients / len(cases) + 1))[:n_clients]
                if n_clients > 1
                else [1]
            )
            sum_cases = sum(cases)
            partition_sizes = [1.0 / sum_cases * case for case in cases]
            self.random_state.shuffle(partition_sizes)

        # create data partitioner.
        self.data_partitioner = DataPartitioner(
            dataset,
            partition_sizes=partition_sizes,
            partition_type=partition_data_conf["distribution"],
            partition_alphas=partition_data_conf["non_iid_alpha"],
            consistent_indices=True,
            random_state=self.random_state,
            graph=self.graph,
            logger=self.logger,
        )
        # create a data partitioner for natural shift test
        _, hist = record_class_distribution(self.data_partitioner.partitions, self.data_partitioner.data.targets)
        if self.data_name == "cifar10" or self.data_name == "imagenet" or self.data_name == "oh":
            self.natural_shift_partitions = partition_by_other_histogram(
                hist,
                self.dataset["natural_shift_test"]
            )
        elif self.data_name == "imagenet1":
            self.natural_shift_partitions = {}
            for d_n in self.natural_shift_list:
                self.natural_shift_partitions[d_n] = partition_by_other_histogram(
                    hist,
                    self.dataset["natural_shift_test"][d_n]
                )

    def intra_client_data_partition_and_create_dataloaders(
        self,
        localdata_id,
        other_ids=None,
        is_in_childworker=False,
        local_train_ratio=0.6,
        batch_size=None,
        shuffle=True,
        display_log=True,
    ):
        """partition clients' data to train, val, test."""
        assert hasattr(self, "data_partitioner")
        batch_size = self.batch_size if batch_size is None else int(batch_size)
        other_ids=random.sample( other_ids, len( other_ids))
        # get the partitioned natural distribution shift dataset.
        log_message = f"Data partition for train (client_id={localdata_id + 1})."
        data_to_load = self.data_partitioner.use(localdata_id)
        if self.data_name == "cifar10"  or self.data_name == "imagenet" or self.data_name == "oh":
            # local_natural_shift = set of local shift (natural)
            local_natural_shift = self.natural_shift_partitions[localdata_id]
            #print(len(local_natural_shift))
            # ooc_natural_shift = set of ooc shift (natural)
            ooc_natural_shifts=[]
            for other_id in other_ids:
                ooc_natural_shifts.append(self.natural_shift_partitions[other_id])
            ooc_natural_shift=ConcatDataset(ooc_natural_shifts) 
            ooc_natural_shift= Subset(
                dataset=ooc_natural_shift,
                indices=self.random_state.choice(len(ooc_natural_shift), int(len(ooc_natural_shift)/len(other_ids)), replace=False),
            )
            print(len(ooc_natural_shift))
            # ooc_test = set of ooc original
            ooc_tests = []
            local_test_ratio = (1 - local_train_ratio) / 2 
            for other_id in other_ids:
                local_data_partitioner = DataPartitioner(
                    self.data_partitioner.use(other_id - 1),
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
                    ooc_tests.append(local_data_partitioner.use(1))
                    ooc_tests.append(local_data_partitioner.use(2))
                elif self.conf.eval_dataset == "test_loader":
                    ooc_tests.append(local_data_partitioner.use(1))
                    ooc_tests.append(local_data_partitioner.use(2))
                  
            ooc_test = ConcatDataset(ooc_tests) 
            ooc_test = Subset(
                dataset=ooc_test,
                indices=self.random_state.choice(len(ooc_test), int(len(ooc_test)/len(other_ids)), replace=False),
            )
            print(len(ooc_test))
        
            # ooc_corr_test = set of ooc shift(corr)
            ooc_corr_test = corr_data.define_corr_data(
                    data=ooc_test,
                    seed=self.conf.corr_seed,
                    severity=self.conf.corr_severity,
            )
            
            # local_corr_test = set of local shift(corr) 
            # 本地测试集和训练集还要耦合，所以在 dataloader解藕
            local_test_ratio = (1 - local_train_ratio) / 2
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
            local_corr_test = corr_data.define_corr_data(
                    data=local_data_partitioner.use(2) if self.conf.eval_dataset == "test_loader" else local_data_partitioner.use(1),
                    seed=self.corr_seed,
                    severity=self.corr_severity,
                )
            _create_dataloader_fn_tr = functools.partial(
                self.create_dataloader, batch_size=batch_size, shuffle=shuffle
                )
           
           
            _create_dataloader_fn = functools.partial(
                self.create_dataloader, batch_size=batch_size, shuffle=shuffle
                )
          

            data_loader_local_tr = _create_dataloader_fn_tr(local_data_partitioner.use(0))
            data_loader_local_val = _create_dataloader_fn(local_data_partitioner.use(1))
            #time_te= Subset(local_data_partitioner.use(2), indices=self.random_state.choice(len(local_data_partitioner.use(2)), 1000, replace=True))
            data_loader_local_te = _create_dataloader_fn(local_data_partitioner.use(2))
            #data_loader_local_te = _create_dataloader_fn(time_te)
            data_loader_ooc_test = _create_dataloader_fn(ooc_test)

            # local shifted
            local_shifts = ConcatDataset([    
                    Subset(local_corr_test, indices=self.random_state.choice(len(local_corr_test), len(local_corr_test), replace=False)),      
                        local_natural_shift,    
                    ])
            data_loader_local_shift=_create_dataloader_fn(local_shifts)

            # ooc shifted
            ooc_shifts = ConcatDataset([    
                   Subset(ooc_corr_test, indices=self.random_state.choice(len(ooc_corr_test), len(ooc_corr_test), replace=False)),      
                        ooc_natural_shift,    
                    ])
            data_loader_ooc_shift=_create_dataloader_fn(ooc_shifts)
            
            # mix
            local_id_test = local_data_partitioner.use(2) if self.conf.eval_dataset == "test_loader" else local_data_partitioner.use(1)
            mix=ConcatDataset([ 
                Subset(ooc_shifts, indices=self.random_state.choice(len(ooc_shifts), int(len(local_natural_shift)/2), replace=False)), 
                Subset(local_shifts, indices=self.random_state.choice(len(local_shifts), int(len(local_natural_shift)/2), replace=False)),           
                Subset( ooc_test, indices=self.random_state.choice(len( ooc_test), int(len(local_natural_shift)/2), replace=False)),           
                Subset( local_id_test, indices=self.random_state.choice(len( local_id_test), int(len(local_natural_shift)/2), replace=False)),           
             ])
            #data_loader_mix=_create_dataloader_fn(mix)
            data_loader_mix=_create_dataloader_fn(mix)
            data_loaders = {
                    "train": data_loader_local_tr,
                    "validation": data_loader_local_val,
                    "test": data_loader_local_te,
                    "corr_test": data_loader_local_shift,
                    #"corr_test": data_loader_local_mixed_test_1,
                    "ooc_test": data_loader_ooc_test,
                    #"ooc_corr_test": data_loader_corr_ooc_test,
                    "ooc_corr_test": data_loader_ooc_shift,
                    #"ooc_corr_test": data_loader_local_mixed_test_2,
                    "natural_shift_test": data_loader_local_tr,
                    #"natural_shift_test": data_loader_my,
                    "mixed_test": data_loader_mix,
                    "num_batches_per_device_per_epoch": len(data_loader_local_tr),
                }    
        
        elif self.data_name == "imagenet1":
            local_natural_shift = {}
            for d_n in self.natural_shift_list:
                local_natural_shift[d_n] = self.natural_shift_partitions[d_n][localdata_id]
            ooc_tests = []
            local_test_ratio = (1 - local_train_ratio) / 2
            for other_id in other_ids:
                local_data_partitioner = DataPartitioner(
                    self.data_partitioner.use(other_id - 1),
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
                    ooc_tests.append(local_data_partitioner.use(1))
                elif self.conf.eval_dataset == "test_loader":
                    ooc_tests.append(local_data_partitioner.use(2))
            ooc_test = ConcatDataset(ooc_tests) 
            ooc_test = Subset(
                dataset=ooc_test,
                indices=self.random_state.choice(len(ooc_test), int(len(ooc_test)/len(other_ids)), replace=False),
            )
            ooc_corr_test = corr_data.define_corr_data(
                    data=ooc_test,
                    seed=self.conf.corr_seed,
                    severity=self.conf.corr_severity,
            )
        else:       
            # get the ooc test, and corrupted ooc test, in case of cifar10.
            ooc_tests = []
            local_test_ratio = (1 - local_train_ratio) / 2
            for other_id in other_ids:
                local_data_partitioner = DataPartitioner(
                    self.data_partitioner.use(other_id - 1),
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
                    ooc_tests.append(local_data_partitioner.use(1))
                elif self.conf.eval_dataset == "test_loader":
                    ooc_tests.append(local_data_partitioner.use(2))
            ooc_test = ConcatDataset(ooc_tests)
            ooc_test = Subset(
                dataset=ooc_test,
                indices=self.random_state.choice(len(ooc_test), int(len(ooc_test)), replace=False),
            )
       
        if display_log:
            self.logger.log(
                f"{log_message}: # of tr batches={len(data_loaders['train'])}, # of validation batches={len(data_loaders['validation']) if data_loaders['validation'] is not None else 'NA'}, # of test batches={len(data_loaders['test']) if data_loaders['test'] is not None else 'NA'}, # of corr_test batches={len(data_loaders['corr_test']) if data_loaders['corr_test'] is not None else 'NA'}, # batch_size={batch_size}"
            )
        return data_loaders

    def create_dataloader(self, dataset, batch_size=None, shuffle=True, sampler=None):
        batch_size = self.batch_size if batch_size is None else int(batch_size)

        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            sampler=sampler,
        )
