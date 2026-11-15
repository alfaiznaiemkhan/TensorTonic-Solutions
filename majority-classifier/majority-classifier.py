import numpy as np

def majority_classifier(y_train: list, X_test: list) -> np.ndarray:
    """
    Returns a one-dimensional NumPy array.
    """
    # Write code here
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test)
    

    classes, first_idx,counts = np.unique(y_train, return_index=True,return_counts=True)
    majority_class = classes[np.argmin(np.where(counts == counts.max(), first_idx, len(y_train)))]


    return np.full(shape=(len(X_test),), fill_value=majority_class,
                   dtype=y_train.dtype) 
    