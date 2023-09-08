from typing import Type
import pytorch_lightning as pl

from lightning.base.system import System
from . import phoneme_recognition



DOWNSTREAMS = {
    "phoneme_recognition": (phoneme_recognition.Expert, phoneme_recognition.PRDataModule),
}



def get_downstream(downstream_name: str) -> Type[System]:
    return DOWNSTREAMS[downstream_name][0]


def get_datamodule(downstream_name: str) -> Type[pl.LightningDataModule]:
    return DOWNSTREAMS[downstream_name][1]
