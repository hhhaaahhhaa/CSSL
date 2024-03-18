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


CONFIG_READER_MAPPING_PHONEME_RECOGNITION = {
    tid: phoneme_recognition.ConfigReader for tid in TID_PHONEME_RECOGNITION
}

# CONFIG_READER_MAPPING_AUDIO_CLASSIFICATION = {
#     tid: audio_classification.Expert for tid in TID_AUDIO_CLASSIFICATION
# }


CONFIG_READER_MAPPING = {
    **CONFIG_READER_MAPPING_PHONEME_RECOGNITION,
}


class AutoConfigReader(object):
    def __init__(self):
        raise EnvironmentError(
            "AutoConfigReader is designed to be instantiated "
            "using the `AutoConfigReader.from_tid(tid)` method."
        )

    @classmethod
    def from_tid(cls, tid: str) -> dict:
        config_reader = CONFIG_READER_MAPPING[tid]()
        return config_reader.from_tid(tid)
