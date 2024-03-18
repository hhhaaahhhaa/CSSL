from typing import List, Dict, Iterator, Tuple, Any, Callable
from torch.utils.data import Dataset, ConcatDataset
from torch.utils.data import RandomSampler, BatchSampler, SequentialSampler
import random
import bisect


class MTLDataset(ConcatDataset):
    """ Extend ConcatDataset with names, and return the name additionally when fetching. """
    def __init__(self, datasets: Dict[str, Dataset]) -> None:
        self.dataset_names = list(datasets.keys())
        super().__init__(list(datasets.values()))

    def __getitem__(self, idx) -> Tuple[str, Any]:
        res = super().__getitem__(idx)
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)

        return (self.dataset_names[dataset_idx], res)


class MTLBatchSampler(object):
    """ Works with MTLDataet. """
    def __init__(
        self,
        mtl_dataset: MTLDataset,
        batch_size: int,
        shuffle: bool=False,
        drop_last: bool=False,
    ) -> None:
        self.names = []
        self.samplers = {}
        self.shuffle = shuffle
        self.length = 0

        offset = 0
        for name, ds in zip(mtl_dataset.dataset_names, mtl_dataset.datasets):
            size = len(ds)
            indices = [x + offset for x in range(size)]
            if shuffle:
                sampler = BatchSampler(RandomSampler(indices), batch_size=batch_size, drop_last=drop_last)
            else:
                sampler = BatchSampler(SequentialSampler(indices), batch_size=batch_size, drop_last=drop_last)
            self.samplers[name] = sampler
            self.names.extend([name] * len(sampler))  # determine which dataset to use for __iter__
            offset += size
            self.length += len(sampler)

    def __iter__(self) -> Iterator[List[int]]:
        if self.shuffle:
            random.shuffle(self.names)
        name_iter = iter(self.names)
        smapler_iters = {name: iter(sampler) for name, sampler in self.samplers.items()}
        while True:
            try:
                name = next(name_iter)
                yield next(smapler_iters[name])
            except StopIteration:
                break

    def __len__(self):
        return self.length


class MTLCollate(object):
    def __init__(self, collate_fns: Dict[str, Callable]) -> None:
        self.collate_fns = collate_fns

    def __call__(self, mtl_data):
        """ mtl_data is a list of tuples [(tid1, data1), (tid2, data2), ...] """
        tid = mtl_data[0][0]  # all tuples should share the same tid
        data = [x[1] for x in mtl_data]
        return (tid, self.collate_fns[tid](data))
