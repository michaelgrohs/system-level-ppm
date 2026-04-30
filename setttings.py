
from __future__ import annotations

# General packages
from typing import Callable, Optional, Union, Literal, Sequence
import os
import random

# Data processing packages
import numpy as np
import pandas as pd

# Models and metrics

import torch

# Types
ArrayLike = Union[np.ndarray, Sequence[float]]
ForecastFn = Callable[[pd.Series, int, Optional[dict]], np.ndarray]
# Model keys
ts_model = Literal["naive", "seasonal_naive", "ets", "sarimax", "ridge", "gru", "nbeats"]


# Default column names
DEFAULT_CASE_COL = "case:concept:name"
DEFAULT_ACT_COL = "concept:name"
DEFAULT_TIME_COL = "time:timestamp"
DEFAULT_RES_COL = "org:resource"

# ===================================
# (1) Basic utilities
# ===================================

def set_global_seed(seed: int, deterministic_torch: bool = True) -> None:
    """
    Sets all seeds to the parameter value for reproducibility.

    - seed: integer seed value to set for all libraries
    - deterministic_torch: if True, also set PyTorch to deterministic mode (may impact performance)
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic_torch:
        # cuDNN / CUDA determinism
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Force deterministic algorithms (may raise if op has no deterministic impl)
        torch.use_deterministic_algorithms(True)

