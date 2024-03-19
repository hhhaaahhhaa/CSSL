from lightning.base.system import BaseSystem
from lightning.system import (
    MTL,
    ONE,
    CFT,
)


SYSTEM_MAPPING: dict[str, BaseSystem] = {
    "MTL-hubert": MTL.hubert.System,
    "ONE-hubert": ONE.hubert.System,
    # "CTrain/hubert": CTrain.hubert.HubertSystem,
    # "CTrain/wav2vec2": CTrain.hubert.HubertSystem,
    # "SlowLearner/hubert": SlowLearner.hubert.HubertSLSystem,
    # "SPU/hubert": SPU.hubert.HubertSPUSystem,
    # "EWC/hubert": EWC.hubert.HubertEWCSystem,
}



class AutoSystem(object):
    r"""
    This is a generic system class that will be instantiated as one of the system classes of the library when
    created with the [`AutoSystem.from_config`] class method.

    This class cannot be instantiated directly using `__init__()` (throws an error).
    """
    def __init__(self):
        raise EnvironmentError(
            "AutoSystem is designed to be instantiated "
            "using the `AutoSystem.from_config(config)` method."
        )

    @classmethod
    def from_config(cls, config) -> BaseSystem:
        system_cls = SYSTEM_MAPPING[config["system_name"]]
        return system_cls(config)
    
    @classmethod
    def load_from_checkpoint(cls, system_name: str, ckpt_path):
        system_cls = SYSTEM_MAPPING[system_name]
        return system_cls.load_from_checkpoint(ckpt_path)
