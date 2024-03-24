from pytorch_lightning.callbacks import Callback
from lightning.task import (
    TID_PHONEME_RECOGNITION,
    TID_AUDIO_CLASSIFICATION,
)
from lightning.task.category import (
    audio_classification,
    mlm,
    phoneme_recognition,
    speaker_identification,
)
from lightning import saver


SAVER_MAPPING_PHONEME_RECOGNITION = {
    tid: phoneme_recognition.Saver for tid in TID_PHONEME_RECOGNITION
}

SAVER_MAPPING_AUDIO_CLASSIFICATION = {
    tid: audio_classification.Saver for tid in TID_AUDIO_CLASSIFICATION
}

SAVER_MAPPING = {
    **SAVER_MAPPING_PHONEME_RECOGNITION,
    **SAVER_MAPPING_AUDIO_CLASSIFICATION,
    "mtl-pr": saver.phoneme_recognition.MTLSaver,
    "mtl-sc": saver.audio_classification.MTLSaver,
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
