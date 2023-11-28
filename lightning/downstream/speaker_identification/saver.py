from typing import Any, Dict, List, Tuple
import os
import pandas as pd
import pytorch_lightning as pl
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from tqdm import tqdm
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers.logger import merge_dicts


CSV_COLUMNS = ["Total Loss"]
COL_SPACE = [len(col) for col in ["200000", "Validation"]+CSV_COLUMNS]  # max step: 200000, longest stage: validation


def set_format(keys: List[str]):
    global CSV_COLUMNS, COL_SPACE
    CSV_COLUMNS = keys
    COL_SPACE = [len(col) for col in ["200000", "Validation"]+CSV_COLUMNS]


class Saver(Callback):

    def __init__(self, data_configs, log_dir, result_dir):
        super().__init__()
        self.data_configs = data_configs
        self.classes = self.data_configs[0]["classes"]

        self.log_dir = log_dir
        self.result_dir = result_dir
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)
        print("Log directory:", self.log_dir)
        print("Result directory:", self.result_dir)

        self.val_loss_dicts = []
        self.val_accs = []
        self.log_loss_dicts = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        loss = outputs['losses']
        output = outputs['output']
        _batch = outputs['_batch']
        step = pl_module.global_step + 1
        if isinstance(pl_module.logger, list):
            assert len(list(pl_module.logger)) == 1
            logger = pl_module.logger[0]
        else:
            logger = pl_module.logger

        # Log message to log.txt and print to stdout
        if step % trainer.log_every_n_steps == 0 and pl_module.local_rank == 0:
            loss_dict = {k: v.item() for k, v in loss.items()}
            set_format(list(loss_dict.keys()))
            loss_dict.update({"Step": step, "Stage": "Training"})
            df = pd.DataFrame([loss_dict], columns=["Step", "Stage"]+CSV_COLUMNS)
            if len(self.log_loss_dicts)==0:
                tqdm.write(df.to_string(header=True, index=False, col_space=COL_SPACE))
            else:
                tqdm.write(df.to_string(header=True, index=False, col_space=COL_SPACE).split('\n')[-1])
            self.log_loss_dicts.append(loss_dict)

            # log classification results
            self.log_text(logger, "Train/GT: " + self.classes[int(_batch["labels"][0])], step)
            self.log_text(logger, "Train/Pred: " + self.classes[int(output[0].argmax())], step)

    def on_validation_epoch_start(self, trainer, pl_module):
        self.val_loss_dicts = []
        self.val_accs = []

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        loss = outputs['losses']
        output = outputs['output']
        _batch = outputs['_batch']
        acc = outputs['acc']
        
        step = pl_module.global_step + 1
        if isinstance(pl_module.logger, list):
            assert len(list(pl_module.logger)) == 1
            logger = pl_module.logger[0]
        else:
            logger = pl_module.logger

        loss_dict = {k: v.item() for k, v in loss.items()}
        self.val_loss_dicts.append(loss_dict)
        self.val_accs.append(acc)

        # Log loss for each sample to csv files
        # self.log_csv("Validation", step, 0, loss_dict)

        # Log classification results to logger
        if batch_idx < 2 and pl_module.local_rank == 0:
            # log asr results
            self.log_text(logger, "Val/GT: " + self.classes[int(_batch["labels"][0])], step)
            self.log_text(logger, "Val/Pred: " + self.classes[int(output[0].argmax())], step)

    def on_validation_epoch_end(self, trainer, pl_module):
        loss_dict = merge_dicts(self.val_loss_dicts)
        step = pl_module.global_step+1

        # Log total loss to log.txt and print to stdout
        loss_dict.update({"Step": step, "Stage": "Validation"})
        # To stdout
        df = pd.DataFrame([loss_dict], columns=["Step", "Stage"]+CSV_COLUMNS)
        if len(self.log_loss_dicts)==0:
            tqdm.write(df.to_string(header=True, index=False, col_space=COL_SPACE))
        else:
            tqdm.write(df.to_string(header=True, index=False, col_space=COL_SPACE).split('\n')[-1])
        # To file
        self.log_loss_dicts.append(loss_dict)
        log_file_path = os.path.join(self.log_dir, 'log.txt')
        df = pd.DataFrame(self.log_loss_dicts, columns=["Step", "Stage"]+CSV_COLUMNS).set_index("Step")
        df.to_csv(log_file_path, mode='a', header=not os.path.exists(log_file_path), index=True)
        # Reset
        self.log_loss_dicts = []

        acc_file_path = os.path.join(self.log_dir, 'acc.txt')
        if len(self.val_accs) > 0:
            with open(acc_file_path, 'a') as f:
                f.write(f"{sum(self.val_accs) / len(self.val_accs)}\n")
    
    def log_csv(self, stage, step, basename, loss_dict):
        if stage in ("Training", "Validation"):
            log_dir = os.path.join(self.log_dir, "csv", stage)
        else:
            log_dir = os.path.join(self.result_dir, "csv", stage)
        os.makedirs(log_dir, exist_ok=True)
        csv_file_path = os.path.join(log_dir, f"{basename}.csv")

        df = pd.DataFrame(loss_dict, columns=CSV_COLUMNS, index=[step])
        df.to_csv(csv_file_path, mode='a', header=not os.path.exists(csv_file_path), index=True, index_label="Step")

    def log_text(self, logger, text, step):
        if isinstance(logger, pl.loggers.CometLogger):
            logger.experiment.log_text(
                text=text,
                step=step,
            )
