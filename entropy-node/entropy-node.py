import numpy as np

def entropy_node(y: list[int]) -> float:
    """
    Returns the Shannon entropy as a Python float.
    """
    # Write code here
    if len(y) == 0:
        return 0.0
        
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / counts.sum()
    
    # Filter out zero probabilities to avoid log2(0) issues, though np.unique already excludes 0 counts
    probabilities = probabilities[probabilities > 0]
    
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return float(entropy)