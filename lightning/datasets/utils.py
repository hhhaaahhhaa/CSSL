import random


class EpisodicInfiniteWrapper:
    def __init__(self, dataset, epoch_length, weights=None):
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
