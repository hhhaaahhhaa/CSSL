import os
import torch.nn as nn
from torch.utils.data import ConcatDataset

from lightning.base.expert import BaseExpert
from .dataset import PRDataset
from .mapper import PRMapper
from .saver import PRSaver
from .collate import PRCollate


class Expert(BaseExpert):
    """
    Fully characterize the task. A task consists of train/val/test datasets and a mapper 
    for the core model to become an expert model.
    Note that task itself can contain multiple datasets if user required.
    keys:
        dataset_config: Data information
        mapper_config: Architecture information
    """
    def __init__(self, config) -> None:
        super().__init__(config)
        self.root = self.config["root"]
        self.dataset_configs = self.config["dataset_configs"]
        self.mapper_config = self.config["mapper_config"]

    def get_train_dataset(self):
        train_datasets = []
        for config in self.dataset_configs:
            if 'train' in config['subsets']:
                filepath = os.path.join(self.root, config['subsets']['train'])
                train_datasets.append(PRDataset(filepath, config))
        return ConcatDataset(train_datasets)

    def get_validation_dataset(self):
        val_datasets = []
        for config in self.dataset_configs:
            if 'val' in config['subsets']:
                filepath = os.path.join(self.root, config['subsets']['val'])
                val_datasets.append(PRDataset(filepath, config))
        return ConcatDataset(val_datasets)

    def get_test_dataset(self):
        test_datasets = []
        for config in self.dataset_configs:
            if 'test' in config['subsets']:
                filepath = os.path.join(self.root, config['subsets']['test'])
                test_datasets.append(PRDataset(filepath, config, is_test=True))
        return ConcatDataset(test_datasets)
    
    def get_train_collate_fn(self):
        return PRCollate(sort=False)

    def get_validation_collate_fn(self):
        return PRCollate(sort=False)

    def get_mapper(self, core: nn.Module):
        return PRMapper(self.mapper_config, core)
    
    def get_saver(self):
        return PRSaver(self.config)
