import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import TensorBoardLogger, CometLogger
from pytorch_lightning.loggers.logger import merge_dicts
from pytorch_lightning.utilities import rank_zero_only
import pandas as pd
from tqdm import tqdm
from collections import defaultdict


class MTLSaver(Callback):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.log_dir = config["output_dir"]["log_dir"]
        self.result_dir = config["output_dir"]["result_dir"]
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)

        self.train_loss_dicts = []
        self.val_loss_dicts = []
        self.val_labels = defaultdict(lambda: defaultdict(list))

    @rank_zero_only
    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs, batch, batch_idx) -> None:
        step = pl_module.global_step + 1
        logger = pl_module.logger
        
        record = outputs['record']
        tid = record["tid"]
        train_loss_dict = {f"Train/{k}": v.item() for k, v in record['losses'].items()}
        self.train_loss_dicts.append(train_loss_dict)
        if step % trainer.log_every_n_steps == 0 and (batch_idx + 1) % self.config["train_config"]["optimizer"]["grad_acc_step"] == 0:
            # log average loss
            avg_train_loss_dict = merge_dicts(self.train_loss_dicts)
            pl_module.log_dict(avg_train_loss_dict, sync_dist=True, batch_size=pl_module.bs)
            tqdm.write(f"Step {step}: {str(avg_train_loss_dict)}")

            # log classification results
            output = record['output']
            labels, _ = output["batch"]
            gt_label = int(labels["labels"][0])
            pred_label = int(output["predictions"][0].argmax())
            
            self.log_text(logger, pl_module.mappers[tid].classes[gt_label], step, f"[{tid}] Train/GT")
            self.log_text(logger, pl_module.mappers[tid].classes[pred_label], step, f"[{tid}] Train/Pred")

            self.train_loss_dicts.clear()

    @rank_zero_only
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        step = pl_module.global_step + 1
        logger = pl_module.logger
        
        record = outputs['record']
        tid = record['tid']
        val_loss_dict = {f"Val/{k}": v.item() for k, v in record['losses'].items()}
        self.val_loss_dicts.append(val_loss_dict)

        output = record['output']
        self.val_labels[tid]["gt"].extend(output["gt"])
        self.val_labels[tid]["pred"].extend(output["pred"])

    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module):
        step = pl_module.global_step + 1
        logger = pl_module.logger

        # log average loss
        avg_val_loss_dict = merge_dicts(self.val_loss_dicts)
        pl_module.log_dict(avg_val_loss_dict, sync_dist=True, batch_size=pl_module.bs)
        tqdm.write(str(avg_val_loss_dict))
        
        for tid in pl_module.mappers:
            # log acc
            acc_file_path = os.path.join(self.log_dir, f'acc-{tid}.txt')
            acc = 0
            assert len(self.val_labels[tid]["gt"]) == len(self.val_labels[tid]["pred"])
            for a, b in zip(self.val_labels[tid]["gt"], self.val_labels[tid]["pred"]):
                if a == b:
                    acc += 1
            acc /= len(self.val_labels[tid]["gt"])
            pl_module.log_dict({f"[{tid}] Val/Acc": acc}, sync_dist=True, batch_size=pl_module.bs)
            with open(acc_file_path, 'a') as f:
                f.write(f"Epoch {pl_module.current_epoch}: {acc * 100:.2f}%\n")

            # log classification results
            for i in range(2):
                self.log_text(logger, self.val_labels[tid]['gt'][i], step, f"[{tid}] Val/GT-{i}")
                self.log_text(logger, self.val_labels[tid]['pred'][i], step, f"[{tid}] Val/Pred-{i}")

        self.val_loss_dicts.clear()
        self.val_labels.clear()
    
    def log_text(self, logger, text, step, tag):
        if isinstance(logger, CometLogger):
            logger.experiment.log_text(
                text=text,
                step=step,
                metadata={"tag": tag}
            )
        elif isinstance(logger, TensorBoardLogger):
            logger.experiment.add_text(tag, text, step)
