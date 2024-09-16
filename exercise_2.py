import numpy as np
from collections import Counter

def lsq_wght(X_train, y_train):

    # Count the number of occurrences of each unique value
    unique_rows, inverse_indices = np.unique(X_train, axis=0, return_inverse=True)
    counts = Counter(inverse_indices)
    
    # Create weights based on counts
    d = np.array([counts[idx] for idx in inverse_indices])
    
    # Add a column of ones to include the intercept term
    X_intercept = np.c_[np.ones((X_train.shape[0], 1)), X_train]  # (n x 2) matrix with intercept
    
    # Create the diagonal weight matrix W
    W = np.diag(d)
    
    # Step 4: Implement Weighted Linear Regression
    X_T_W = X_intercept.T @ W
    beta_weighted = np.linalg.inv(X_T_W @ X_intercept) @ (X_T_W @ y_train)

    return beta_weighted, d
