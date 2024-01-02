import os
import json
import yaml


def debug():
    root = "task_config/debug"
    os.makedirs(root, exist_ok=True)

    tasks = [
        "lightning/systems/CTrain/hubert/data_config/LibriSpeech",
        "lightning/systems/CTrain/hubert/data_config/ESC50",
        "lightning/systems/CTrain/hubert/data_config/UrbanSound8K",
    ]
    with open(f"{root}/config.yaml", 'w') as f:
        yaml.dump({"tasks": tasks}, f)
    
    # create tid sequence
    tid_seq, saving_steps = [], []
    for i in range(len(tasks)):
        tid_seq.extend([i]* 50)
        saving_steps.append((i + 1) * 50)
    task_boundaries = [0] + saving_steps
    
    with open(f"{root}/tid_seq.json", 'w') as f:
        json.dump(tid_seq, f)

    with open(f"{root}/saving_steps.json", 'w') as f:
        json.dump(saving_steps, f)

    with open(f"{root}/task_boundaries.json", 'w') as f:
        json.dump(task_boundaries, f)


def exp1a():
    root = "task_config/exp1a-seq"
    os.makedirs(root, exist_ok=True)

    tasks = [
        "lightning/systems/CTrain/hubert/data_config/AISHELL-3",
        "lightning/systems/CTrain/hubert/data_config/CSMSC",
        "lightning/systems/CTrain/hubert/data_config/LJSpeech",
        "lightning/systems/CTrain/hubert/data_config/LibriTTS",
    ]
    with open(f"{root}/config.yaml", 'w') as f:
        yaml.dump({"tasks": tasks}, f)
    
    # create tid sequence
    tid_seq, saving_steps = [], []
    for i in range(len(tasks)):
        tid_seq.extend([i]* 40000)
        saving_steps.append((i + 1) * 40000)
    task_boundaries = [0] + saving_steps
    
    with open(f"{root}/tid_seq.json", 'w') as f:
        json.dump(tid_seq, f)

    with open(f"{root}/saving_steps.json", 'w') as f:
        json.dump(saving_steps, f)

    with open(f"{root}/task_boundaries.json", 'w') as f:
        json.dump(task_boundaries, f)


def exp1b():
    root = "task_config/exp1b-seq"
    os.makedirs(root, exist_ok=True)

    tasks = [
        "lightning/systems/CTrain/hubert/data_config/ESC50",
        "lightning/systems/CTrain/hubert/data_config/Urban8K",
        "lightning/systems/CTrain/hubert/data_config/TAU2019UAS",
    ]
    with open(f"{root}/config.yaml", 'w') as f:
        yaml.dump({"tasks": tasks}, f)
    
    # create tid sequence
    tid_seq, saving_steps = [], []
    for i in range(len(tasks)):
        tid_seq.extend([i]* 40000)
        saving_steps.append((i + 1) * 40000)
    task_boundaries = [0] + saving_steps
    
    with open(f"{root}/tid_seq.json", 'w') as f:
        json.dump(tid_seq, f)

    with open(f"{root}/saving_steps.json", 'w') as f:
        json.dump(saving_steps, f)

    with open(f"{root}/task_boundaries.json", 'w') as f:
        json.dump(task_boundaries, f)


if __name__ == "__main__":
    debug()
    exp1a()
    exp1b()
