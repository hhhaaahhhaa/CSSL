from pytorch_lightning.callbacks import Callback

from lightning.base.system import System


class SavingScheduleCallback(Callback):
    def __init__(self, saving_steps: list[int]) -> None:
        super().__init__()
        self.saving_steps = saving_steps

    def on_train_batch_end(self, trainer, pl_module: System, outputs, batch, batch_idx) -> None:
        if pl_module.global_step + 1 in self.saving_steps:
            trainer.save_checkpoint(f"{pl_module.ckpt_dir}/step={pl_module.global_step}.ckpt")
