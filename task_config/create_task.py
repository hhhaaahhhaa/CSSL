import os
import json
import yaml


def exp1a():
    root = "./exp1a-seq"
    os.makedirs(root, exist_ok=True)

    tasks = [
        "lightning/systems/CTrain/hubert/data_config/LJSpeech",
        "lightning/systems/CTrain/hubert/data_config/ESC50",
        "lightning/systems/CTrain/hubert/data_config/CSMSC",
    ]
    with open(f"{root}/config.yaml", 'w') as f:
        yaml.dump({"tasks": tasks}, f)
    
    # create tid sequence
    tid_seq, saving_steps = [], []
    for i in range(len(tasks)):
        tid_seq.extend([i]* 40000)
        saving_steps.append((i + 1) * 40000)
    
    with open(f"{root}/tid_seq.json", 'w') as f:
        json.dump(tid_seq, f)

    with open(f"{root}/saving_steps.json", 'w') as f:
        json.dump(saving_steps, f)


if __name__ == "__main__":
    exp1a()
