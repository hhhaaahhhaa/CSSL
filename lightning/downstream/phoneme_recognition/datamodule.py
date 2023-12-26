import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, ConcatDataset

import Define
from lightning.datasets.utils import EpisodicInfiniteWrapper
from .parser import DataParser
from .dataset import PRDataset
from .collate import PRCollate
from .config_reader import ConfigReader


class PRDataModule(pl.LightningDataModule):
    """
    Train: PRDataset + PRCollate.
    Val: PRDataset + PRCollate.
    """
    def __init__(self, data_configs, model_config, train_config, algorithm_config, log_dir, result_dir, dataset_cls=PRDataset):
        super().__init__()
        self.data_configs = [ConfigReader.read(x) for x in data_configs]
        self.model_config = model_config
        self.train_config = train_config
        self.algorithm_config = algorithm_config

        self.log_dir = log_dir
        self.result_dir = result_dir
        self.val_step = self.train_config["step"]["val_step"]

        self.collate = PRCollate()

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

        # if stage in (None, 'test', 'predict'):
        #     self.test_datasets = [
        #         TextDataset(
        #             data_config['subsets']['test'],
        #             data_config
        #         ) for data_config in self.data_configs if 'test' in data_config['subsets']
        #     ]
        #     self.test_dataset = ConcatDataset(self.test_datasets)
        #     self._test_setup()

    def _train_setup(self):
        self.batch_size = self.train_config["optimizer"]["batch_size"]
        self.train_dataset = ConcatDataset(self.train_datasets)
        
    def _validation_setup(self):
        self.val_dataset = ConcatDataset(self.val_datasets)

    def _test_setup(self):
        pass

    def train_dataloader(self):
        """Training dataloader, not modified for multiple dataloaders."""
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=Define.MAX_WORKERS,
            collate_fn=self.collate.collate_fn(),
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

    # def test_dataloader(self):
    #     """Test dataloader"""
    #     self.test_loader = DataLoader(
    #         self.test_dataset,
    #         batch_size=self.batch_size//torch.cuda.device_count(),
    #         shuffle=False,
    #         collate_fn=self.collate2.collate_fn(False, re_id=False),
    #     )
    #     return self.test_loader
