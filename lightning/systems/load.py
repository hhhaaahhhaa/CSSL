from typing import Type
import pytorch_lightning as pl

from lightning.base.system import System
from . import s3prl
from . import CTrain
from . import SlowLearner
from . import SPU
from . import EWC


SYSTEM_CTRAIN = {
    "CTrain/hubert": (CTrain.hubert.HubertSystem, CTrain.hubert.DataModule),
    "CTrain/wav2vec2": (CTrain.hubert.HubertSystem, CTrain.hubert.DataModule),
}


SYSTEM_SL = {
    "SlowLearner/hubert": (SlowLearner.hubert.HubertSLSystem, CTrain.hubert.DataModule),
    "SlowLearner/wav2vec2": (SlowLearner.hubert.HubertSLSystem, CTrain.hubert.DataModule),
}


SYSTEM_SPU = {
    "SPU/hubert": (SPU.hubert.HubertSPUSystem, SPU.hubert.DataModule),
}


SYSTEM_EWC = {
    "EWC/hubert": (EWC.hubert.HubertEWCSystem, SPU.hubert.DataModule),
}


SYSTEM = {
    **SYSTEM_CTRAIN,
    **SYSTEM_SL,
    **SYSTEM_SPU,
    **SYSTEM_EWC,
}


def get_system(system_name: str) -> Type[System]:
    return SYSTEM[system_name][0]


def get_datamodule(system_name: str) -> Type[pl.LightningDataModule]:
    return SYSTEM[system_name][1]


def load_system(system_name, ckpt_file):
    if system_name not in ["hubert", "wav2vec2"]:
        assert ckpt_file is not None
        system = get_system(system_name)
        upstream = system.load_from_checkpoint(ckpt_file)
        return upstream
    
    # s3prl upstreams
    return s3prl.S3PRLWrapper(system_name)
