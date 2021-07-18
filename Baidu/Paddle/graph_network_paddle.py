

import pgl
import paddle
import paddle.nn as nn
import numpy as np

from typing import Callable

def build_mlp(
    hidden_size: int, num_hidden_layers:int,output_size:int):
    """Build an MLP"""
    
