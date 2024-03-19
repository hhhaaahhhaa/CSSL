import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import TensorBoardLogger, CometLogger
from pytorch_lightning.loggers.logger import merge_dicts
from pytorch_lightning.utilities import rank_zero_only
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import jiwer


# CSV_COLUMNS = ["Total Loss"]
# COL_SPACE = [len(col) for col in ["200000", "Validation"]+CSV_COLUMNS]  # max step: 200000, longest stage: validation


# def set_format(keys):
#     global CSV_COLUMNS, COL_SPACE
#     CSV_COLUMNS = keys
#     COL_SPACE = [len(col) for col in ["200000", "Validation"]+CSV_COLUMNS]


class Saver(Callback):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.log_dir = config["output_dir"]["log_dir"]
        self.result_dir = config["output_dir"]["result_dir"]
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.result_dir, exist_ok=True)

        self.train_loss_dicts = []
        self.val_loss_dicts = []
        self.val_transcriptions = defaultdict(list)

    @rank_zero_only
    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs, batch, batch_idx) -> None:
        step = pl_module.global_step + 1
        logger = pl_module.logger
        
        record = outputs['record']
        train_loss_dict = {f"Train/{k}": v.item() for k, v in record['losses'].items()}
        self.train_loss_dicts.append(train_loss_dict)
        if step % trainer.log_every_n_steps == 0 and (batch_idx + 1) % self.config["train_config"]["optimizer"]["grad_acc_step"] == 0:
            # log average loss
            avg_train_loss_dict = merge_dicts(self.train_loss_dicts)
            pl_module.log_dict(avg_train_loss_dict, sync_dist=True, batch_size=pl_module.bs)
            tqdm.write(f"Step {step}: {str(avg_train_loss_dict)}")

            # set_format(list(loss_dict.keys()))
            # loss_dict.update({"Step": step, "Stage": "Training"})
            # df = pd.DataFrame([loss_dict], columns=["Step", "Stage"]+CSV_COLUMNS)
            # if len(self.train_loss_dicts)==0:
            #     tqdm.write(df.to_string(header=True, index=False, col_space=COL_SPACE))
            # else:
            #     tqdm.write(df.to_string(header=True, index=False, col_space=COL_SPACE).split('\n')[-1])

            # if len(self.train_loss_dicts)==1:
            #     tqdm.write(df.to_string(header=True, index=False, col_space=COL_SPACE))
            # tqdm.write(df.to_string(header=True, index=False, col_space=COL_SPACE).split('\n')[-1])

            # log phoneme recognition results
            output = record['output']
            beam_search_results = pl_module.beam_search_decoder(output["emissions"], output["emission_lengths"])
            labels, _ = output["batch"]
            gt_label = labels["texts"][0][:labels["text_lens"][0]].cpu()
            gt_transcript = pl_module.beam_search_decoder.idxs_to_tokens(gt_label)
            gt_transcript = " ".join([p for p in gt_transcript if p != "|"])

            beams = beam_search_results[0]
            pred_transcript = pl_module.beam_search_decoder.idxs_to_tokens(beams[0].tokens)
            pred_transcript = " ".join([p for p in pred_transcript if p != "|"])
            
            self.log_text(logger, gt_transcript, step, "Train/GT")
            self.log_text(logger, pred_transcript, step, "Train/Pred")

            self.train_loss_dicts.clear()

    @rank_zero_only
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        step = pl_module.global_step + 1
        logger = pl_module.logger
        
        record = outputs['record']
        val_loss_dict = {f"Val/{k}": v.item() for k, v in record['losses'].items()}
        self.val_loss_dicts.append(val_loss_dict)

        output = record['output']
        self.val_transcriptions["gt"].extend(output["gt"])
        self.val_transcriptions["pred"].extend(output["pred"])

        # Log loss for each sample to csv files
        # self.log_csv("Validation", step, 0, loss_dict)

    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module):
        step = pl_module.global_step + 1
        logger = pl_module.logger

        # log average loss
        avg_val_loss_dict = merge_dicts(self.val_loss_dicts)
        pl_module.log_dict(avg_val_loss_dict, sync_dist=True, batch_size=pl_module.bs)
        tqdm.write(str(avg_val_loss_dict))
        
        # Log total loss to log.txt and print to stdout
        # loss_dict.update({"Step": step, "Stage": "Validation"})
        # # To stdout
        # df = pd.DataFrame([loss_dict], columns=["Step", "Stage"]+CSV_COLUMNS)
        # if len(self.log_loss_dicts)==0:
        #     tqdm.write(df.to_string(header=True, index=False, col_space=COL_SPACE))
        # else:
        #     tqdm.write(df.to_string(header=True, index=False, col_space=COL_SPACE).split('\n')[-1])
        # # To file
        # self.log_loss_dicts.append(loss_dict)
        # log_file_path = os.path.join(self.log_dir, 'log.txt')
        # df = pd.DataFrame(self.log_loss_dicts, columns=["Step", "Stage"]+CSV_COLUMNS).set_index("Step")
        # df.to_csv(log_file_path, mode='a', header=not os.path.exists(log_file_path), index=True)
        # # Reset
        # self.log_loss_dicts = []

        # log PER
        per_file_path = os.path.join(self.log_dir, 'per.txt')
        per = jiwer.wer(self.val_transcriptions["gt"], self.val_transcriptions["pred"])
        pl_module.log_dict({f"Val/PER": per}, sync_dist=True, batch_size=pl_module.bs)
        with open(per_file_path, 'a') as f:
            f.write(f"Epoch {pl_module.current_epoch}: {per * 100:.2f}%\n")

        # log asr results
        for i in range(2):
            self.log_text(logger, self.val_transcriptions['gt'][i], step, f"Val/GT-{i}")
            self.log_text(logger, self.val_transcriptions['pred'][i], step, f"Val/Pred-{i}")

        self.val_loss_dicts.clear()
        self.val_transcriptions.clear()
    
    # def log_csv(self, stage, step, basename, loss_dict):
    #     if stage in ("Training", "Validation"):
    #         log_dir = os.path.join(self.log_dir, "csv", stage)
    #     else:
    #         log_dir = os.path.join(self.result_dir, "csv", stage)
    #     os.makedirs(log_dir, exist_ok=True)
    #     csv_file_path = os.path.join(log_dir, f"{basename}.csv")

    #     df = pd.DataFrame(loss_dict, columns=CSV_COLUMNS, index=[step])
    #     df.to_csv(csv_file_path, mode='a', header=not os.path.exists(csv_file_path), index=True, index_label="Step")

    def log_text(self, logger, text, step, tag):
        if isinstance(logger, CometLogger):
            logger.experiment.log_text(
                text=text,
                step=step,
                metadata={"tag": tag}
            )
        elif isinstance(logger, TensorBoardLogger):
            logger.experiment.add_text(tag, text, step)