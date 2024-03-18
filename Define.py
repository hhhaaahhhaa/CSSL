import os
import torch


# Machine settings
LOCAL = True
CUDA_LAUNCH_BLOCKING = False
MAX_WORKERS = 4
if LOCAL:
    CUDA_LAUNCH_BLOCKING = True  # Always crash on my PC if false
    MAX_WORKERS = 2
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"  # https://stackoverflow.com/questions/73747731/runtimeerror-cuda-out-of-memory-how-can-i-set-max-split-size-mb
torch.set_float32_matmul_precision("high")  # https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision

# Logger
LOGGER = "tb"  # none/comet/tb

# Global variables
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEBUG = False
DATAPARSERS = {}
CTC_DECODERS = {}
