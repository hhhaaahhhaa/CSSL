from typing import Callable
import os
from tqdm import tqdm

from lightning.base.evaluator import BaseEvaluator
from .config_reader import ConfigReader
from .expert import Expert


class Evaluator(BaseEvaluator):
    def __init__(self, config) -> None:
        super().__init__(config)
        config_reader = ConfigReader({"tid": self.config["tid"]})
        expert = Expert(config=config_reader.task_config)
        self.dataset = expert.get_test_dataset()
        self.logdir = f"{self.config['logdir']}/evaluator/{self.config['tid']}"
        os.makedirs(self.logdir, exist_ok=True)

    def run(self, func: Callable):
        gt_labels, pred_labels = [], []
        for idx, data in tqdm(enumerate(self.dataset), total=len(self.dataset)):
            gt = data["classes"]
            pred = func(data["wav"])
            gt_labels.append(gt)
            pred_labels.append(pred)
            # print(pred)
            # input()
        acc = 0
        for a, b in zip(gt_labels, pred_labels):
            if a == b:
                acc += 1
        acc /= len(gt_labels)
        with open(f"{self.logdir}/results.txt", 'w') as f:
            f.write(f"Acc: {acc * 100:.2f}%\n")
