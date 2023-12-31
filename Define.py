import torch


# Machine settings
LOCAL = False
CUDA_LAUNCH_BLOCKING = False
MAX_WORKERS = 4
if LOCAL:
    CUDA_LAUNCH_BLOCKING = True  # TODO: Always crash on my PC if false
    MAX_WORKERS = 2

# Logger
LOGGER = "tb"  # none/comet/tb

# Global variables
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEBUG = False
DATAPARSERS = {}
CTC_DECODERS = {}
