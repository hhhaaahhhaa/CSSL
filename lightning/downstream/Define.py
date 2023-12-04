import os
import torch
import yaml


# Machine settings
LOCAL = True
CUDA_LAUNCH_BLOCKING = False
MAX_WORKERS = 4
if LOCAL:
    CUDA_LAUNCH_BLOCKING = True  # Always crash on my PC if false
    MAX_WORKERS = 2

# Global variables
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEBUG = False

# Task variables
DOWNSTREAM_TASKS = yaml.load(open(f"{os.path.dirname(__file__)}/task.yaml", "r"), Loader=yaml.FullLoader)
