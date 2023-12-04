from typing import Type

from dlhlp_lib.parsers.Interfaces import BasePreprocessor

# from .TAT import TATPreprocessor
# from .TAT_TTS import TATTTSPreprocessor
from .librispeech import LibriSpeechPreprocessor
from .libritts import LibriTTSPreprocessor
from .ljspeech import LJSpeechPreprocessor
from .aishell3 import AISHELL3Preprocessor
from .css10 import CSS10Preprocessor
from .csmsc import CSMSCPreprocessor
from .kss import KSSPreprocessor
from .jsut import JSUTPreprocessor
from .m_ailabs import MAILABSPreprocessor


PREPROCESSORS = {
    "LibriSpeech": LibriSpeechPreprocessor,
    "LJSpeech": LJSpeechPreprocessor,
    "LibriTTS": LibriTTSPreprocessor,
    "AISHELL-3": AISHELL3Preprocessor,
    "CSS10": CSS10Preprocessor,
    "CSMSC": CSMSCPreprocessor,
    "KSS": KSSPreprocessor,
    "JSUT": JSUTPreprocessor,
    "M-AILABS": MAILABSPreprocessor,
    # "TAT": TATPreprocessor,
    # "TATTTS": TATTTSPreprocessor,
}


def get_preprocessor(tag: str) -> Type[BasePreprocessor]:
    return PREPROCESSORS[tag]
