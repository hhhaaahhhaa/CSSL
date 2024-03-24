from lightning.base.evaluator import BaseEvaluator
from lightning.task import (
    TID_PHONEME_RECOGNITION,
    TID_AUDIO_CLASSIFICATION,
)
from lightning.task.category import (
    phoneme_recognition,
    audio_classification,
)


EVALUATOR_MAPPING_PHONEME_RECOGNITION = {
    tid: phoneme_recognition.Evaluator for tid in TID_PHONEME_RECOGNITION
}

EVALUATOR_MAPPING_AUDIO_CLASSIFICATION = {
    tid: audio_classification.Evaluator for tid in TID_AUDIO_CLASSIFICATION
}


EVALUATOR_MAPPING = {
    **EVALUATOR_MAPPING_PHONEME_RECOGNITION,
    **EVALUATOR_MAPPING_AUDIO_CLASSIFICATION,
}


class AutoEvaluator(object):
    def __init__(self):
        raise EnvironmentError(
            "AutoEvaluator is designed to be instantiated "
            "using the `AutoEvaluator.from_tid(tid, logdir)` method."
        )

    @classmethod
    def from_tid(cls, tid: str, logdir: str, *args, **kwargs) -> BaseEvaluator:
        evaluator_cls = EVALUATOR_MAPPING[tid]
        config = {"tid": tid, "logdir": logdir}
        return evaluator_cls(config, *args, **kwargs)
