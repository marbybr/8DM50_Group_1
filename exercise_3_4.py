import numpy as np
import scipy

# Function to compute distances
compute_distance = lambda entry, matrix: scipy.spatial.distance.cdist(entry.reshape(1, -1), matrix).reshape(-1,)

# Function to get indices of nearest neighbors
compute_neighbors = lambda distances, k: np.argsort(distances)[:k]

# Function to get most common nearest neighbors
# Make it so that it can both be used for classification and regression
def nearest_neighbor(y, idx, kind="classification"):

    # Classification part
    if kind == "classification":
        # Only get labels of nearest neighbors
        arr = y[idx]

        # Find the unique values and their counts
        unique_values, counts = np.unique(arr, return_counts=True)

        # Find the index of the maximum count
        most_common_index = np.argmax(counts)

        # Get the most common value
        most_common_value = unique_values[most_common_index]

        return most_common_value

    # Regression part
    elif kind == "regression":
        # Only get labels of nearest neighbors
        arr = y[idx]

        # Return the average of the nearest neighbors
        return np.mean(arr)
    
    else:
        raise ValueError(f"{kind} not in [\"classification\", \"regression\"]")
    
# Finally, a function to compute the nearest neighbors for the entire dataset
def knn_predict(entry, X_train, y_train, k, kind="classification"):

    # Compute distances
    dists = compute_distance(entry, X_train)

    # Get indices of nearest neighbors
    idx = compute_neighbors(dists, k)

    # Return prediction
    pred = nearest_neighbor(y_train, idx, kind=kind)
    return pred

def nearest_neighbors(X_test, X_train, y_train, k, kind="classification"):
    """Computes the nearest neighbors for given data

    ### Args:
        - `data (np.ndarray)`: Input data for which to predict the target values
        - `X_train (np.ndarray)`: Data used to get nearest neighbors
        - `y_train (np.ndarray)`: Labels of the nearest neighbors
        - `k` (int): Number of nearest neighbors
        - `kind` (str): \"classification\" or \"regression\"
    """

    # Make predictions
    predictions = np.asarray([knn_predict(entry, X_train, y_train, k, kind=kind) for entry in X_test])
    return predictions

# # accuracy = lambda y_test, y_pred: 
accuracy = lambda y_test, y_pred: len(np.where(y_pred.flatten() == y_test.flatten())[0]) / len(y_test)
