import os
import yaml
from typing import Type
import pytorch_lightning as pl

from lightning.base.system import System
from . import Define
from . import phoneme_recognition
from . import sound_classification
from . import speaker_identification


DOWNSTREAMS = {
    "phoneme_recognition": (phoneme_recognition.Expert, phoneme_recognition.PRDataModule),
    "sound_classification": (sound_classification.Expert, sound_classification.ClassificationDataModule),
    "speaker_identification": (speaker_identification.Expert, speaker_identification.ClassificationDataModule),
}


def get_downstream(downstream_name: str) -> Type[System]:
    try:
        return DOWNSTREAMS[downstream_name][0]
    except:
        raise ValueError(f"Can not found downstream {downstream_name}.")


def get_datamodule(downstream_name: str) -> Type[pl.LightningDataModule]:
    try:
        return DOWNSTREAMS[downstream_name][1]
    except:
        raise ValueError(f"Can not found downstream {downstream_name}.")


def get_configs(downstream_name: str, task_name: str):
    downstream_dir = f"lightning/downstream/{downstream_name}"
    fullname = Define.DOWNSTREAM_TASKS[downstream_name][task_name]

    model_config_path = f"{downstream_dir}/config/model.yaml"
    train_config_path = f"{downstream_dir}/config/train.yaml"
    algorithm_config_path = f"{downstream_dir}/config/algorithm.yaml"

    if os.path.exists(f"{downstream_dir}/config/{fullname}/model.yaml"):
        model_config_path = f"{downstream_dir}/config/{fullname}/model.yaml"
    if os.path.exists(f"{downstream_dir}/config/{fullname}/train.yaml"):
        train_config_path = f"{downstream_dir}/config/{fullname}/train.yaml"
    if os.path.exists(f"{downstream_dir}/config/{fullname}/algorithm.yaml"):
        algorithm_config_path = f"{downstream_dir}/config/{fullname}/algorithm.yaml"

    model_config = yaml.load(open(model_config_path, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(train_config_path, "r"), Loader=yaml.FullLoader)
    algorithm_config = yaml.load(open(algorithm_config_path, "r"), Loader=yaml.FullLoader)
    tmp = f"lightning/downstream/{downstream_name}/data_config/{fullname}"
    data_configs = [tmp]

    return data_configs, model_config, train_config, algorithm_config
