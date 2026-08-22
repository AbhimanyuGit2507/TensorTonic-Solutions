import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float = 0.1, steps: int = 1000) -> tuple[np.ndarray, float]:
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    w,b=np.zeros(X.shape[1]),0

    for _ in range(steps):
        pred = np.dot(X,w) + b
        model = _sigmoid(pred)
        Xt = np.transpose(X)
    
        dw = np.dot(Xt, (model - y)) / len(X)
        db = np.mean(model - y)
    
        w-=lr*dw
        b-=lr*db

    return w,float(b)