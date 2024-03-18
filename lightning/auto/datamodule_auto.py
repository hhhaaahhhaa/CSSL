import pytorch_lightning as pl
from lightning.systems import (
    MTL,
    ONE,
    CTrain,
    SPU,
)


DATAMODULE_MAPPING = {
    "MTL-hubert": MTL.datamodule.MTLDataModule,
    "ONE-hubert": ONE.datamodule.ONEDataModule,
    "SL-hubert": MTL.datamodule.MTLDataModule,
    "CTrain/hubert": CTrain.hubert.DataModule,
    "CTrain/wav2vec2": CTrain.hubert.DataModule,
    "SlowLearner/hubert": SPU.hubert.DataModule,
    "SPU/hubert": SPU.hubert.DataModule,
    "EWC/hubert": SPU.hubert.DataModule,
}



class AutoDatamodule(object):
    r"""
    This is a generic datamodule class that will be instantiated as one of the datamodule classes of the library when
    created with the [`AutoDatamodule.from_config`] class method.

    This class cannot be instantiated directly using `__init__()` (throws an error).
    """
    def __init__(self):
        raise EnvironmentError(
            "AutoDatamodule is designed to be instantiated "
            "using the `AutoDatamodule.from_config(config)` method."
        )

    @classmethod
    def from_config(cls, config) -> pl.LightningDataModule:
        datamodule_cls = DATAMODULE_MAPPING[config["system_name"]]
        return datamodule_cls(config)
