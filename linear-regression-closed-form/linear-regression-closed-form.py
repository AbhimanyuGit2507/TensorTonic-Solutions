import numpy as np

def linear_regression_closed_form(X: list, y: list) -> list:
    """
    Returns the optimal weight vector as a list.
    """
    Xt = np.transpose(X)

    w = np.dot(np.dot(np.linalg.inv(np.dot(Xt,X)),Xt),y)

    return w