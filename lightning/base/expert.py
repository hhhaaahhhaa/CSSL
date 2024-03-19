from typing import Callable
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pytorch_lightning as pl


class BaseMapper(pl.LightningModule):
    def __init__(self, config, core: nn.Module, *args, **kwargs) -> None:
        super().__init__()
        self.config = config
        self.core = core

    def build_optimized_model(self):
        """ Return modules to be updated in training loop, core should be excluded. """
        raise NotImplementedError

    def training_step(self, batch, batch_idx, record={}) -> torch.Tensor:
        """ Return train loss and log information into record. """
        raise NotImplementedError
    
    def validation_step(self, batch, batch_idx, record={}):
        """ Return validation loss and log information into record. """
        raise NotImplementedError


class BaseExpert(object):
    def __init__(self, config, *args, **kwargs) -> None:
        self.config = config

    def get_train_dataset(self) -> Dataset:
        raise NotImplementedError
    
    def get_validation_dataset(self) -> Dataset:
        raise NotImplementedError
    
    def get_test_dataset(self) -> Dataset:
        raise NotImplementedError
    
    def get_train_collate_fn(self) -> Callable:
        raise NotImplementedError
    
    def get_validation_collate_fn(self) -> Callable:
        raise NotImplementedError
    
    def get_mapper(self) -> BaseMapper:
        raise NotImplementedError
