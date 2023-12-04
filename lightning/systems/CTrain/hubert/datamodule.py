import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, ConcatDataset

import Define
from lightning.systems.task_reader import TaskSequenceConfig, PlainConfigAdapter
from lightning.datasets.utils import EpisodicInfiniteWrapper, TaskSequenceWrapper, batch_collate_wrapper
from .dataset import CodeDataset
from .collate import Collate
from .config_reader import ConfigReader


class DataModule(pl.LightningDataModule):
    """
    Train: CodeDataset + Collate.
    Val: CodeDataset + Collate.
    """
    def __init__(self, data_configs, model_config, train_config, algorithm_config, log_dir, result_dir, dataset_cls=CodeDataset):
        super().__init__()
        # Compatible loading
        if "task_config" in data_configs[0]:
            self.task_config = TaskSequenceConfig(data_configs[0])
        else:
            self.task_config = PlainConfigAdapter(data_configs)
        self.data_configs = [ConfigReader.read(x) for x in self.task_config.get_tasks()]

        self.model_config = model_config
        self.train_config = train_config
        self.algorithm_config = algorithm_config

        self.log_dir = log_dir
        self.result_dir = result_dir
        self.val_step = self.train_config["step"]["val_step"]

        self.collate = Collate()

        self.dataset_cls = dataset_cls

    def setup(self, stage=None):
        if stage in (None, 'fit', 'validate'):
            self.train_datasets = [
                self.dataset_cls(
                    data_config['subsets']['train'],
                    data_config
                ) for data_config in self.data_configs if 'train' in data_config['subsets']
            ]
            self.val_datasets = [
                self.dataset_cls(
                    data_config['subsets']['val'],
                    data_config
                ) for data_config in self.data_configs if 'val' in data_config['subsets']
            ]
            self._train_setup()
            self._validation_setup()

        if stage in ('test', 'predict'):
            raise NotImplementedError

    def _train_setup(self):
        self.batch_size = self.train_config["optimizer"]["batch_size"]
        grad_acc_step = self.train_config["optimizer"].get("grad_acc_step", 1)
        info = self.task_config.get_info()
        if info["tid_seq"] is None:  # Default iid version
            self.train_dataset = ConcatDataset(self.train_datasets)
            self.train_dataset = EpisodicInfiniteWrapper(self.train_dataset, self.val_step*self.batch_size)
        else:
            self.train_dataset = TaskSequenceWrapper(info["tid_seq"], self.train_datasets, self.batch_size, grad_acc_step)

    def _validation_setup(self):
        self.val_dataset = self.val_datasets[0]  # dummy, validation during CSSL training is not informative, need to apply downstream task.

    def _test_setup(self):
        pass

    def train_dataloader(self):
        """Training dataloader, not modified for multiple dataloaders."""
        if getattr(self.train_dataset, "is_batched", False):
            collate_fn = batch_collate_wrapper(self.collate.collate_fn())
            batch_size = 1
        else:
            collate_fn = self.collate.collate_fn()
            batch_size = self.batch_size
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=False,  # Shuffle in wrapper
            num_workers=Define.MAX_WORKERS,
            collate_fn=collate_fn,
        )
        return self.train_loader

    def val_dataloader(self):
        """Validation dataloader, not modified for multiple dataloaders."""
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=self.collate.collate_fn(),
        )
        return self.val_loader

    def test_dataloader(self):
        """Test dataloader"""
        raise NotImplementedError
    