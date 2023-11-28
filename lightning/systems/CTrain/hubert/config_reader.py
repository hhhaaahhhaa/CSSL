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
        if "unit_type" not in config:
            config["unit_type"] = config["name"]

        assert "unit_name" in config
        assert "n_symbols" in config
        
        return config
