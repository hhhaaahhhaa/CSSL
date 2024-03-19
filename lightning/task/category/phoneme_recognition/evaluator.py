from typing import Callable
import os
import jiwer
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
        gt_transcriptions, pred_transcriptions = [], []
        for idx, data in tqdm(enumerate(self.dataset), total=len(self.dataset)):
            gt = data["phoneme"]
            pred = func(data["wav"])
            gt_transcriptions.append(gt)
            pred_transcriptions.append(pred)
            # print(pred)
            # input()
        wer = jiwer.wer(gt_transcriptions, pred_transcriptions)
        with open(f"{self.logdir}/results.txt", 'w') as f:
            f.write(f"PER: {wer * 100:.2f}%\n")
