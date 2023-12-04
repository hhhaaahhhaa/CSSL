import numpy as np
from torch.utils.data import Dataset
import random


class EpisodicInfiniteWrapper:
    def __init__(self, dataset: Dataset, epoch_length: int, weights=None):
        self.dataset = dataset
        self.epoch_length = epoch_length
        self.idxs = list(range(len(self.dataset)))
        self.ws = [1] * len(self.dataset)
        if weights is not None:
            assert isinstance(weights, list) and len(weights) == len(dataset)
            self.ws = weights

    def __getitem__(self, idx):
        idx = random.choices(self.idxs, weights=self.ws, k=1)[0]
        return self.dataset[idx]

    def __len__(self):
        return self.epoch_length


class TaskSequenceWrapper:
    def __init__(self, tid_seq: list[int], datasets: list[Dataset], batch_size: int):
        self.tid_seq = tid_seq
        self.epoch_length = len(tid_seq)
        self.datasets = datasets
        self.bs = batch_size

    @property
    def is_batched(self):
        return True
    
    def __getitem__(self, idx):
        dataset = self.datasets[self.tid_seq[idx]]
        idxs = np.random.randint(len(dataset), size=self.bs)
        return [dataset[t] for t in idxs]

    def __len__(self):
        return self.epoch_length


def batch_collate_wrapper(collate_func):
    return lambda data_batch: collate_func(data_batch[0])
