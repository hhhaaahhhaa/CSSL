from lightning.base.expert import BaseExpert
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


EXPERT_MAPPING_PHONEME_RECOGNITION = {
    tid: phoneme_recognition.Expert for tid in TID_PHONEME_RECOGNITION
}

EXPERT_MAPPING_AUDIO_CLASSIFICATION = {
    tid: audio_classification.Expert for tid in TID_AUDIO_CLASSIFICATION
}


EXPERT_MAPPING = {
    **EXPERT_MAPPING_PHONEME_RECOGNITION,
    **EXPERT_MAPPING_AUDIO_CLASSIFICATION,
}


class AutoExpert(object):
    r"""
    This is a generic expert class that will be instantiated as one of the expert classes of the library when
    created with the [`AutoExpert.from_config`] class method.

    This class cannot be instantiated directly using `__init__()` (throws an error).
    """
    def __init__(self):
        raise EnvironmentError(
            "AutoExpert is designed to be instantiated "
            "using the `AutoExpert.from_config(config)` method."
        )

    @classmethod
    def from_config(cls, config) -> BaseExpert:
        expert_cls = EXPERT_MAPPING[config["tid"]]
        return expert_cls(config)
