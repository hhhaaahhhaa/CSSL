import abc


class ITaskBoundary(metaclass=abc.ABCMeta):
    def is_task_boundary(self) -> bool:
        raise NotImplementedError
    
    def task_init(self) -> None:
        raise NotImplementedError

    def task_end(self) -> None:
        raise NotImplementedError
