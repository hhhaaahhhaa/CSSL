"""
Load core, fix core, train mapper only.
This is a part of SSL evaluation pipeline.
"""
import argparse
import os

import comet_ml
import pytorch_lightning as pl
from pytorch_lightning.profilers import SimpleProfiler
from pytorch_lightning.loggers import TensorBoardLogger, CometLogger
import torch
import yaml
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import Define
from lightning.auto import AutoSystem, AutoDatamodule, AutoConfigReader, AutoSaver


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


TRAINER_CONFIG = {
    "accelerator": "gpu" if torch.cuda.is_available() else None,
    "strategy": "ddp_find_unused_parameters_true",  # multigpu should use ddp
    "use_distributed_sampler": False,  # Add this to ensure no shuffle on dataset (Designed task Sequence in CL will broke.)
}


def create_config(args) -> dict:
    """ Create a dictionary for full configuration """
    res = {
        "exp_name": args.exp_name,
        "system_name": args.system_name,
        "model_config": None,
        "train_config": None,
        "algorithm_config": None,
    }
    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    res.update(config)

    tid = args.tid
    res["task_configs"] = {tid: AutoConfigReader.from_tid(tid)}

    # Useful for debugging
    if Define.DEBUG:
        res["train_config"]["optimizer"]["batch_size"] = 1
        res["train_config"]["step"]["log_step"] = 10

    return res


def main(args):
    print("Prepare training ...")
    config = create_config(args)

    # create output directories and dump the full config
    exp_root = f"exp/{args.system_name}/{args.exp_name}"
    os.makedirs(exp_root, exist_ok=True)
    config["output_dir"] = {
        "log_dir": f"{exp_root}/log",
        "result_dir": f"{exp_root}/result",
        "ckpt_dir": f"{exp_root}/ckpt"
    }
    with open(f"{exp_root}/config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(config, f)
    if Define.DEBUG:
        print("Config prepared.")
        input()

    # Init logger
    if Define.LOGGER == "comet":
        from config.comet import COMET_CONFIG
        logger = CometLogger(
            save_dir=config["output_dir"]["log_dir"],
            experiment_name=args.exp_name,
            **COMET_CONFIG
        )
        logger.log_hyperparams(config)
    elif Define.LOGGER == "tb":
        logger = TensorBoardLogger(save_dir=config["output_dir"]["log_dir"], name="")
    else:
        pass

    datamodule = AutoDatamodule.from_config(config)
    if Define.DEBUG:
        print("Data module prepared.")
        input()

    system = AutoSystem.from_config(config)
    if Define.DEBUG:
        print("System module prepared.")
        input()

    savers = []
    saver_names = config.get("saver_names", [])
    savers = [AutoSaver.from_config(config, saver_name=saver_name) for saver_name in saver_names]
    if Define.DEBUG:
        print("Saver modules prepared.")
        input()

    # Training
    train_config = config["train_config"]
    trainer_training_config = {
        "profiler": SimpleProfiler(dirpath=exp_root, filename="profile"),
        'max_epochs': train_config["step"]["max_epochs"],
        'log_every_n_steps': train_config["step"]["log_step"],
    }
    if system.automatic_optimization:
        trainer_training_config.update({
            'gradient_clip_val': train_config["optimizer"]["grad_clip_thresh"],
            'accumulate_grad_batches': train_config["optimizer"]["grad_acc_step"],
        })
    trainer_training_config = {**trainer_training_config, **TRAINER_CONFIG}
    # Useful for debugging
    if Define.DEBUG:
        trainer_training_config.update({
            "limit_train_batches": 100,
            "limit_val_batches": 10,
            "max_epochs": 3
        })

    pl.seed_everything(43, True)
    print("========================== Start Training! ==========================")
    print("Task ID: ", args.tid)
    print("Exp name: ", config["exp_name"])
    print("System name: ", config["system_name"])
    print("Core checkpoint path: ", args.core_ckpt_file)
    print("Log directory: ", config["output_dir"]["log_dir"])
    print("Result directory: ", config["output_dir"]["result_dir"])
    print("Checkpoint directory: ", config["output_dir"]["ckpt_dir"])

    if args.core_ckpt_file is not None and args.ckpt_file is None:
        system.load_core_checkpoint(args.core_ckpt_file)
    trainer = pl.Trainer(logger=logger, callbacks=savers, **trainer_training_config)
    trainer.fit(system, datamodule=datamodule, ckpt_path=args.ckpt_file)


if __name__ == "__main__":
    """
    Usage:
        python mapper_train.py -n ONE-hubert -t en0 -n unnamed --config config/mapper_train.yaml 
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--system_name", type=str, help="system identifier", default="MTL-hubert")
    parser.add_argument(
        "-n", "--exp_name", type=str, help="experiment name, default is algorithm's name", default="unnamed"
    )
    parser.add_argument(
        "-t", "--tid", type=str, help="single task name", default="en0"
    )
    parser.add_argument(
        "-c", "--core_ckpt_file", type=str, help="core checkpoint file name",
        default=None,
    )
    parser.add_argument(
        "--config", type=str, help="config file name",
    )
    parser.add_argument(
        "--logger", type=str, help="output result path",
        default="tb",
    )
    parser.add_argument(
        "--ckpt_file", type=str, help="resume checkpoint file name",
        default=None,
    )
    parser.add_argument("--debug", action="store_true", default=False)
    
    args = parser.parse_args()
    Define.DEBUG = args.debug
    Define.LOGGER = args.logger
   
    main(args)
