import os
import yaml
import copy


class ConfigReader(object):
    def __init__(self):
        self.default_config = yaml.load(open(
            "lightning/task/category/phoneme_recognition/default_config.yaml", "r"), Loader=yaml.FullLoader)
    
    def from_tid(self, tid: str):
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

        res = copy.deepcopy(self.default_config["task_config"])
        res.update(task_config)

        # add tokens for phoneme recognition
        from lightning.text.define import LANG_ID2SYMBOLS
        res["mapper_config"]["tokens"] = LANG_ID2SYMBOLS[res["lang_id"]]

        return res
