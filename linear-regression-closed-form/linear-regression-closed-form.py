import numpy as np

def linear_regression_closed_form(X: list, y: list) -> list:
    """
    Returns the optimal weight vector as a list.
    """
    X=np.array(X)
    s = np.linalg.inv(X.T @ X) @ X.T @ y
    return s.tolist()