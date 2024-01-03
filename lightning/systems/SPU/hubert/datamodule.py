from torch.utils.data import DataLoader

import Define
from lightning.datasets.utils import InfiniteWrapper
from lightning.systems.CTrain import hubert


class DataModule(hubert.DataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def task_dataloader(self, idx, batch_size):
        ds = self.train_datasets[idx]
        self.train_loader = DataLoader(
            InfiniteWrapper(ds, shuffle=True),
            batch_size=batch_size,
            num_workers=Define.MAX_WORKERS,
            collate_fn=self.collate.collate_fn(),
        )
        return self.train_loader
