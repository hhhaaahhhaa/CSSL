from typing import Type
import pytorch_lightning as pl

from lightning.base.system import System
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
