from torch.utils.data import DataLoader

import Define
from lightning.datasets.utils import InfiniteWrapper
from lightning.systems.CTrain import hubert


class DataModule(hubert.DataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def task_dataloader(self, idx, batch_size, infinite=True):
        ds = self.train_datasets[idx]
        if infinite:
            ds = InfiniteWrapper(ds, shuffle=True)
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            num_workers=Define.MAX_WORKERS,
            collate_fn=self.collate.collate_fn(),
        )
        return loader
