import torch


# Machine settings
LOCAL = True
CUDA_LAUNCH_BLOCKING = False
MAX_WORKERS = 4
if LOCAL:
    CUDA_LAUNCH_BLOCKING = True  # TODO: Always crash on my PC if false
    MAX_WORKERS = 2

# Global variables
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEBUG = False
