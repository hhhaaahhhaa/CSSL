import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from lightning.base.expert import BaseExpert
from lightning.auto import AutoExpert
from lightning.datasets.multitask import MTLDataset, MTLBatchSampler, MTLCollate
import Define


class MTLDataModule(pl.LightningDataModule):
    """
    Concat multiple datasets from different tasks and random shuffle.
    Ensure that every batch comes from the same dataset.
    Validation also concats all validation datasets.
    """

    experts: dict[str, BaseExpert]

    def __init__(self, config):
        super().__init__()
        self.task_configs = config["task_configs"]
        self.train_config = config["train_config"]

        # build experts
        self.experts = {}
        for tid, task_config in self.task_configs.items():
            self.experts[tid] = AutoExpert.from_config(task_config)

        # training config
        self.batch_size = self.train_config["optimizer"]["batch_size"]

    def setup(self, stage=None):
        if stage in (None, 'fit', 'validate'):
            self.train_datasets, self.val_datasets = {}, {}
            self.train_collate_fns, self.val_collate_fns = {}, {}
            for tid, expert in self.experts.items():
                self.train_datasets[tid] = expert.get_train_dataset()
                self.val_datasets[tid] = expert.get_validation_dataset()
                self.train_collate_fns[tid] = expert.get_train_collate_fn()
                self.val_collate_fns[tid] = expert.get_validation_collate_fn()
            
            self._train_setup()
            self._validation_setup()

        if stage in ('test', 'predict'):
            raise NotImplementedError

    def _train_setup(self):
        self.train_dataset = MTLDataset(self.train_datasets)
        self.train_sampler = MTLBatchSampler(
            mtl_dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

    def _validation_setup(self):
        self.val_dataset = MTLDataset(self.val_datasets)
        self.val_sampler = MTLBatchSampler(
            mtl_dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=True,
        )

    def _test_setup(self):
        pass

    def train_dataloader(self):
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_sampler=self.train_sampler,
            num_workers=Define.MAX_WORKERS,
            collate_fn=MTLCollate(self.train_collate_fns)
        )
        return self.train_loader

    def val_dataloader(self):
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_sampler=self.val_sampler,
            num_workers=Define.MAX_WORKERS,
            collate_fn=MTLCollate(self.val_collate_fns)
        )
        return self.val_loader

    def test_dataloader(self):
        raise NotImplementedError
