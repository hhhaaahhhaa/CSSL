from typing import Callable


class BaseEvaluator(object):
    def __init__(self, config, *args, **kwargs) -> None:
        self.config = config

    def run(self, func: Callable):
        raise NotImplementedError
