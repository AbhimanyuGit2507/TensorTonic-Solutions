import numpy as np
import math 

def sigmoid(x: list | float) -> np.ndarray | float:
    """
    Vectorized sigmoid function.
    """
    x=np.array(x)
    s = 1/(1+(math.e ** -x))
    return s
    