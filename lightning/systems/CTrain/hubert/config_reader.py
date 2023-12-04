import yaml


class ConfigReader(object):
    def __init__(self):
        pass
    
    @staticmethod
    def read(root):
        config = yaml.load(open(f"{root}/config.yaml", "r"), Loader=yaml.FullLoader)

        if "lang_id" not in config:
            config["lang_id"] = "en"
        for k in config['subsets']:
            config['subsets'][k] = f"{root}/{config['subsets'][k]}"
        if "uid" not in config:
            config["uid"] = config["name"] + ":" + config["unit_name"]

        assert "unit_name" in config
        assert "n_symbols" in config
        assert "uid" in config
        
        return config
