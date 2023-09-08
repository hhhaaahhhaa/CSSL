import yaml


class ConfigReader(object):
    def __init__(self):
        pass
    
    @staticmethod
    def read(root):
        config = yaml.load(open(f"{root}/config.yaml", "r"), Loader=yaml.FullLoader)

        config["data_dir"] = "lightning/downstream/phoneme_recognition/" + config["data_dir"]
        if "lang_id" not in config:
            config["lang_id"] = "en"
        for k in config['subsets']:
            config['subsets'][k] = f"{root}/{config['subsets'][k]}"

        assert "unit_name" in config
        assert "n_symbols" in config
        
        return config
