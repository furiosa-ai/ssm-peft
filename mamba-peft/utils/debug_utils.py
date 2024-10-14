import os
import random
import numpy as np


def enable_deterministic():
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"

    import torch
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False

    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)