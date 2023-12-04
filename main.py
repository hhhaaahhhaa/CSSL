import argparse
import os

import comet_ml
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, CometLogger
import torch
import yaml
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import Define
from lightning.systems import get_system, get_datamodule
from lightning.systems.task_reader import TaskSequenceConfig


quiet = False
if quiet:
    # NOTSET/DEBUG/INFO/WARNING/ERROR/CRITICAL
    os.environ["COMET_LOGGING_CONSOLE"] = "ERROR"
    import warnings
    warnings.filterwarnings("ignore")
    import logging
    # configure logging at the root level of lightning
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
if Define.CUDA_LAUNCH_BLOCKING:
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"


TRAINER_CONFIG = {
    "accelerator": "gpu" if torch.cuda.is_available() else None,
    "strategy": "ddp_find_unused_parameters_true",  # multigpu should use ddp
    "profiler": 'simple',
    "use_distributed_sampler": False,  # Add this to ensure no shuffle on dataset (Designed task Sequence in CL will broke.)
}


def load_configs(args):
    downstream_dir = f"lightning/systems/{args.system}"
    model_config = yaml.load(open(f"{downstream_dir}/config/model.yaml", "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(f"{downstream_dir}/config/train.yaml", "r"), Loader=yaml.FullLoader)
    algorithm_config = yaml.load(open(f"{downstream_dir}/config/algorithm.yaml", "r"), Loader=yaml.FullLoader)
    
    # Compatible loading
    if "task_config" in args.task_config[0]:
        meta_config = TaskSequenceConfig(args.task_config[0])
        n = meta_config.get_info()["training"]["total_step"]
        if n != -1:
            train_config["step"]["total_step"] = n
    
    data_configs = args.task_config

    # Useful for debugging
    if Define.DEBUG:
        train_config["optimizer"]["batch_size"] = 1

    return data_configs, model_config, train_config, algorithm_config


def main(args):
    print("Prepare training ...")
    downstream_dir = f"lightning/systems/{args.system}"

    # load configs
    data_configs, model_config, train_config, algorithm_config = load_configs(args)

    # Init logger
    if Define.LOGGER == "comet":
        from config.comet import COMET_CONFIG
        comet_logger = CometLogger(
            save_dir=f"{downstream_dir}/exp/log",
            experiment_key=args.exp_key,
            experiment_name=args.exp_name,
            **COMET_CONFIG
        )
        comet_logger.log_hyperparams({
            "data_config": data_configs,
            "model_config": model_config,
            "train_config": train_config,
            "algorithm_config": algorithm_config,
        })
        loggers = [comet_logger]
        log_dir = f"{downstream_dir}/exp/log/{comet_logger.version}"
        result_dir = f"{downstream_dir}/exp/{comet_logger.version}/result"
        ckpt_dir = f"{downstream_dir}/exp/{comet_logger.version}/ckpt"
    elif Define.LOGGER == "tb":
        log_dir = f"{downstream_dir}/exp/{args.exp_name}/log"
        result_dir = f"{downstream_dir}/exp/{args.exp_name}/result"
        ckpt_dir = f"{downstream_dir}/exp/{args.exp_name}/ckpt"
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(result_dir, exist_ok=True)
        os.makedirs(ckpt_dir, exist_ok=True)
        tb_logger = TensorBoardLogger(save_dir=log_dir)
        loggers = [tb_logger]
    else:
        pass

    datamodule = get_datamodule(args.system)(
        data_configs, model_config, train_config, algorithm_config, log_dir, result_dir,
    )
    if Define.DEBUG:
        print("Data module prepared.")
        input()

    # Init system
    system = get_system(args.system)
    if args.pretrain_path is None:
        model = system(
            data_configs, model_config, train_config, algorithm_config,
            log_dir, result_dir, ckpt_dir,
        )
    else:
        print("Load from checkpoint...")
        model = system.load_from_checkpoint(
            args.pretrain_path,
            data_configs=data_configs, model_config=model_config, train_config=train_config, algorithm_config=algorithm_config,
            log_dir=log_dir, result_dir=result_dir, ckpt_dir=ckpt_dir
        )
    if Define.DEBUG:
        print("System module prepared.")
        input()

    # Training
    trainer_training_config = {
        'max_steps': train_config["step"]["total_step"],
        'log_every_n_steps': train_config["step"]["log_step"],
        'val_check_interval': train_config["step"]["val_step"],
        'check_val_every_n_epoch': None,
        'gradient_clip_val': train_config["optimizer"]["grad_clip_thresh"],
        'accumulate_grad_batches': train_config["optimizer"]["grad_acc_step"],
    }

    pl.seed_everything(43, True)
    print("========================== Start Training! ==========================")
    # Useful for debugging
    if Define.DEBUG:
        TRAINER_CONFIG.update({
            "limit_train_batches": 200,
            "limit_val_batches": 50,
            "max_epochs": 3
        })
    trainer = pl.Trainer(logger=loggers, **TRAINER_CONFIG, **trainer_training_config)
    trainer.fit(model, datamodule=datamodule, ckpt_path=args.ckpt_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("system", type=str, help="system identifier")
    parser.add_argument(
        "-n", "--exp_name", type=str, help="experiment name, default is algorithm's name",
    )
    parser.add_argument(
        "-t", "--task_config", type=str, nargs='+', help="path to data config directory",
    )
    parser.add_argument(
        "-e", "--exp_key", type=str, help="experiment key (comet)",
        default=None,
    )
    parser.add_argument(
        "-c", "--ckpt_file", type=str, help="ckpt file name",
        default=None,
    )
    parser.add_argument(
        "-pre", "--pretrain_path", type=str, help="pretrained model path",
        default=None,
    )
    parser.add_argument(
        "--logger", type=str, help="output result path",
        default="tb",
    )
    parser.add_argument("--debug", action="store_true", default=False)
    
    args = parser.parse_args()
    Define.DEBUG = args.debug
    Define.LOGGER = args.logger
   
    main(args)
