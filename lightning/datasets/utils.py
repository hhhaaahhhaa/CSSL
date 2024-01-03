from typing import Iterator
import numpy as np
from torch.utils.data import Dataset, IterableDataset
import random


class EpisodicInfiniteWrapper(Dataset):
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


class InfiniteWrapper(IterableDataset):
    def __init__(self, dataset: Dataset, shuffle=False):
        assert len(dataset) > 0
        self.dataset = dataset
        self.length = len(dataset)
        self.shuffle = shuffle
        self.__gen = self.__create_gen()

    def __create_gen(self):
        order = list(range(self.length))
        if self.shuffle:
            random.shuffle(order)
        idx = 0
        while True:
            yield self.dataset[order[idx]]
            idx += 1
            if idx == len(order):
                if self.shuffle:
                    random.shuffle(order)
                idx = 0
                
    def __iter__(self) -> Iterator:
        return self.__gen

    def __next__(self) -> int:
        return next(self.__gen)


class TaskSequenceWrapper(Dataset):
    def __init__(self, tid_seq: list[int], datasets: list[Dataset], batch_size: int, grad_acc_step=1):
        if grad_acc_step > 1:
            self.tid_seq = np.array(tid_seq).repeat(grad_acc_step)
        else:
            self.tid_seq = tid_seq
        self.epoch_length = len(self.tid_seq)
        self.datasets = [InfiniteWrapper(ds, shuffle=True) for ds in datasets]
        self.bs = batch_size

    @property
    def is_batched(self):
        return True
    
    def __getitem__(self, idx):
        # print("getitem:", idx, self.tid_seq[idx])
        dataset = self.datasets[self.tid_seq[idx]]

        res = []
        for _ in range(self.bs):
            res.append(next(dataset))
        return res

    def __len__(self):
        return self.epoch_length


def batch_collate_wrapper(collate_func):
    return lambda data_batch: collate_func(data_batch[0])
