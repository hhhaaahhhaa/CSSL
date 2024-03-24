import os
import yaml
import copy
import json

from lightning.base.config_reader import BaseConfigReader


class ConfigReader(BaseConfigReader):
    def __init__(self, config):
        super().__init__(config)
        self.default_config = yaml.load(open(
            "lightning/task/category/audio_classification/default_config.yaml", "r"), Loader=yaml.FullLoader)

        tid = config["tid"]
        root = f"lightning/task/{tid}"
        task_config = yaml.load(open(f"{root}/config.yaml", "r"), Loader=yaml.FullLoader)

        # basic config
        task_config["tid"] = tid
        task_config["root"] = root

        # parse dataset config
        assert len(task_config["dataset_config_paths"]) > 0, "No dataset"
        task_config["dataset_configs"] = []
        for path in task_config["dataset_config_paths"]:
            dataset_config = yaml.load(open(os.path.normpath(os.path.join(root, path)), "r"), Loader=yaml.FullLoader)
            tmp = copy.deepcopy(self.default_config["dataset_config"])
            tmp.update(dataset_config)
            tmp["data_dir"] = os.path.normpath(os.path.join(root, tmp["data_dir"]))
            task_config["dataset_configs"].append(tmp)

        self.task_config = copy.deepcopy(self.default_config["task_config"])
        self.task_config.update(task_config)

        # add classes (all datasets should share same classes)
        first_data_dir = task_config["dataset_configs"][0]["data_dir"]
        with open(f"{first_data_dir}/classes.json", 'r') as f:
            self.task_config["mapper_config"]["classes"] = json.load(f)

        self.train_config = self.default_config["train_config"]
