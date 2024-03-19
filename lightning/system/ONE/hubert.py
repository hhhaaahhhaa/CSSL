import torch
from pytorch_lightning.callbacks import ModelCheckpoint

from .. import MTL


class System(MTL.hubert.System):
    """ Naive single task learning with hubert base. """

    def __init__(self, config):
        super().__init__(config)

    def build_configs(self) -> None:
        super().build_configs()
        assert len(self.task_configs) == 1
        for tid, task_config in self.task_configs.items():
            self.tid = tid          
    
    def build_saver(self) -> list:
        checkpoint = ModelCheckpoint(
            dirpath=self.config["output_dir"]["ckpt_dir"],
            filename='{epoch}',
            monitor="Val/Total Loss", mode="min",
            save_top_k=-1
        )
        saver = self.experts[self.tid].get_saver()  # default saver class for the task
        return [checkpoint, saver]
    
    def load_core_checkpoint(self, ckpt_file: str):
        self.core.load_state_dict(torch.load(ckpt_file, map_location='cpu'))
