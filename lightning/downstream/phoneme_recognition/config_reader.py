import yaml


class ConfigReader(object):
    def __init__(self):
        pass
    
    @staticmethod
    def read(root):
        config = yaml.load(open(f"{root}/config.yaml", "r"), Loader=yaml.FullLoader)

        config["data_dir"] = "lightning/downstream/phoneme_recognition/" + config["data_dir"]

        for k in config['subsets']:
            config['subsets'][k] = f"{root}/{config['subsets'][k]}"
        # if "symbol_id" not in config:
        #     if "n_symbols" in config:
        #         config["symbol_id"] = config["unit_name"]
        #         config["use_real_phoneme"] = False
        #     else:
        #         config["symbol_id"] = config["lang_id"]
        #         config["use_real_phoneme"] = True

        return config
