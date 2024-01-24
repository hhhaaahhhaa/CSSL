import os
import yaml
import json
import copy


class BaseConfig(object):

    _tasks: list[str]
    _info: dict

    def __init__(self, *args, **kwargs) -> None:
        self._tasks = []
        self._info = {}

    def get_tasks(self, *args, **kwargs):
        return copy.deepcopy(self._tasks)
    
    def get_info(self, *args, **kwargs):
        return copy.deepcopy(self._info)


class PlainConfig(BaseConfig):
    def __init__(self, roots: list[str]):
        super().__init__()
        self._tasks = roots
        self._info = {}


class TaskSequenceConfig(BaseConfig):
    def __init__(self, root: str):
        super().__init__()
        self.root = root
        self.raw_config = yaml.load(open(f"{self.root}/config.yaml", "r"), Loader=yaml.FullLoader)
        self._tasks = self.raw_config["tasks"]
        try:
            self._load_task_seq()
        except:
            self.log("Task sequence loading failed.")
            raise

    def _load_task_seq(self):
        self._info["tid_seq"] = None
        self._info["training"] = {
            "total_step": -1,
            "saving_steps": [],
            "task_boundaries": [],
        }
        if os.path.isfile(f"{self.root}/tid_seq.json"):
            with open(f"{self.root}/tid_seq.json", 'r') as f:
                tid_seq = json.load(f)
                assert max(tid_seq) < len(self._tasks)
                self._info["training"]["total_step"] = len(tid_seq)
                self._info["tid_seq"] = tid_seq
        
        if os.path.isfile(f"{self.root}/saving_steps.json"):
            with open(f"{self.root}/saving_steps.json", 'r') as f:
                self._info["training"]["saving_steps"] = json.load(f)
        
        if os.path.isfile(f"{self.root}/task_boundaries.json"):
            with open(f"{self.root}/task_boundaries.json", 'r') as f:
                self._info["training"]["task_boundaries"] = json.load(f)

    def log(self, msg):
        print(f"[TaskSequenceConfig]: ", msg)


class PlainConfigAdapter(BaseConfig):
    """ Adapt PlainConfig to TaskSequenceConfig """
    def __init__(self, roots: list[str]):
        super().__init__()
        self._tasks = roots
        self._info["tid_seq"] = None
        self._info["training"] = {
            "total_step": -1,
            "saving_steps": [],
            "task_boundaries": [],
        }
