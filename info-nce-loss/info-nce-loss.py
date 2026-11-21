import numpy as np

def info_nce_loss(Z1: list, Z2: list, temperature: float = 0.1) -> float:
    """
    Returns the loss as a float.
    """
    # Write code here
    
    Z1 = np.asarray(Z1)
    Z2 = np.asarray(Z2)

    logits = Z1 @ Z2.T / temperature
    logit_max = np.max(logits, axis=1, keepdims=True)
    log_exp = np.exp(logits - logit_max)
    shifted = logits - logit_max - np.log(np.sum(log_exp, axis=-1, keepdims=True))
    pos_log = np.diag(shifted)
    loss = -np.mean(pos_log)
    return float(loss)