"""
Run task evaluation.
"""
import os
import gc
import torch
import yaml

from lightning.auto import AutoEvaluator, AutoSystem


GENERAL = ["en0", "sid1"]
LANGUAGE = ["en2", "zh2", "ko", "ru"]
AUDIO = ["sc2", "sc3", "genre", "tech"]


def task_eval(tid, ckpt_path, exp_root=None):   
    if exp_root is None:
        exp_root = os.path.dirname(os.path.dirname(ckpt_path))  # default format
    
    evaluator = AutoEvaluator.from_tid(tid, logdir=exp_root)
    print(f"Load model from {ckpt_path}...")
    exp_config = yaml.load(open(f"{exp_root}/config.yaml", "r"), Loader=yaml.FullLoader)
    system = AutoSystem.load_from_checkpoint(
        system_name=exp_config["system_name"],
        ckpt_path=ckpt_path
    )
    system.eval()
    system.cuda()

    # multi-pr
    inference_func = system.mappers[tid].inference

    # run
    print("========================== Start Evaluation! ==========================")
    print("Task ID: ", tid)
    print("Exp root: ", exp_root)
    print("System name: ", exp_config["system_name"])
    print("Checkpoint path: ", ckpt_path)
    evaluator.run(inference_func)

    system.cpu()
    gc.collect()


if __name__ == "__main__":
    for tid in LANGUAGE:
        task_eval(tid, "exp/MTL-hubert/unnamed/ckpt/epoch=2.ckpt")
