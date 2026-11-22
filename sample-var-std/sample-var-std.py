import numpy as np

def sample_var_std(x: list) -> dict:
    """
    Returns a dictionary with variance and standard_deviation.
    """
    # Write code here
    x = np.asarray(x)
    centered = x - np.mean(x)
    s2 = (1 /(x.size - 1) * np.sum(centered ** 2))
    s = np.sqrt(s2)
    return {"variance": float(s2), "standard_deviation": float(s)}