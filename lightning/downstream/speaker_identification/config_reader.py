import yaml
import json


class ConfigReader(object):
    def __init__(self):
        pass
    
    @staticmethod
    def read(root):
        config = yaml.load(open(f"{root}/config.yaml", "r"), Loader=yaml.FullLoader)

        config["data_dir"] = "lightning/downstream/speaker_identification/" + config["data_dir"]

        for k in config['subsets']:
            config['subsets'][k] = f"{root}/{config['subsets'][k]}"
        with open(f"{config['data_dir']}/speakers.json", 'r') as f:
            config["classes"] = json.load(f)

        return config
