from torch.utils.data import DataLoader

import Define
from lightning.systems.CTrain import hubert


class DataModule(hubert.DataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def task_dataloader(self, idx, batch_size):
        print("Task: ", idx)
        ds = self.train_datasets[idx]
        self.train_loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=Define.MAX_WORKERS,
            collate_fn=self.collate.collate_fn(),
        )
        return self.train_loader
