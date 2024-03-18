from pytorch_lightning.callbacks import Callback
from lightning import saver


SAVER_MAPPING = {
    "mtl-pr": saver.phoneme_recognition.MTLSaver
}


class AutoSaver(object):
    def __init__(self):
        raise EnvironmentError(
            "AutoSaver is designed to be instantiated "
            "using the `AutoSaver.from_config(config)` method."
        )

    @classmethod
    def from_config(cls, config, **kwargs) -> Callback:
        config.update(kwargs)
        saver_cls = SAVER_MAPPING[config["saver_name"]]
        return saver_cls(config)
