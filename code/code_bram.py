import numpy as np
from sklearn.datasets import load_diabetes, load_breast_cancer
import scipy
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import norm
import os

# Load datasets
diabetes = load_diabetes()
breast_cancer = load_breast_cancer()

#### Exercise 1: Linear Regression ####

# add subfolder that contains all the function implementations
# to the system path so we can import them
import sys
sys.path.append('code/')

# the actual implementation is in linear_regression.py,
# here we will just use it to fit a model
from linear_regression import *

# load the dataset
# same as before, but now we use all features
X_train = diabetes.data[:300, :]
y_train = diabetes.target[:300, np.newaxis]
X_test = diabetes.data[300:, :]
y_test = diabetes.target[300:, np.newaxis]
# X_train = diabetes.data[:300, np.newaxis, 3]
# y_train = diabetes.target[:300, np.newaxis]
# X_test = diabetes.data[300:, np.newaxis, 3]
# y_test = diabetes.target[300:, np.newaxis]

# Obtain coefficients
beta = lsq(X_train, y_train)

predict = lambda X, beta: np.matmul(X, beta[1:]) + beta[0] # Lambda function for prediction

# Make predictions
y_pred = predict(X_test, beta)

# Get a mean squared error
mean_squared_error = lambda y, y_pred: np.mean((y - y_pred)**2)
mse_ex1 = mean_squared_error(y_test, y_pred)

# Print mean squared error
print(f"Mean Squared Error exercise 1: {mse_ex1}")

#### Exercise 2: Weighted Linear Regression ####
# Concatenate X and y for unique sample identification
X_y_train = np.hstack([X_train, y_train])

# Get unique samples and their counts
unique_data, counts = np.unique(X_y_train, axis=0, return_counts=True)

# Separate the unique X and y
X_unique = unique_data[:, :-1]  # All columns except the last one
y_unique = unique_data[:, -1:]  # Only the last column

def weighted_lsq(X, y, weights):
    """
    Calculate the weighted least squares solution for linear regression.
    :param X: Design matrix (features), shape (n_samples, n_features)
    :param y: Response vector, shape (n_samples, 1)
    :param weights: Weights vector, shape (n_samples,)
    :return: Coefficient vector, shape (n_features + 1, 1)
    """
    # Add intercept term
    X = np.hstack([np.ones((X.shape[0], 1)), X])
    
    # Create a diagonal weight matrix
    W = np.diag(weights)
    
    # Calculate beta using the weighted least squares formula
    XTWX = X.T @ W @ X
    XTWy = X.T @ W @ y
    beta = np.linalg.inv(XTWX) @ XTWy
    return beta

# Fit the weighted linear regression model
weights = counts  # Use counts as weights
beta_weighted = weighted_lsq(X_unique, y_unique, weights)

# Define the prediction function with an intercept
predict_weighted = lambda X, beta: np.matmul(np.hstack([np.ones((X.shape[0], 1)), X]), beta)

# Make predictions on the test set
y_pred_weighted = predict_weighted(X_test, beta_weighted)

# Calculate mean squared error
mse_weighted = mean_squared_error(y_test, y_pred_weighted)

print(f"Mean Squared Error (Weighted) exercise 2: {mse_weighted:.4f}")

#### Exercise 3: k-NN classification ####
# Get data from breast_cancer dataset
X_train = breast_cancer.data[:400, np.newaxis, 3] # Slightly larger than diabetes because more rows
y_train = breast_cancer.target[:400, np.newaxis]
X_test = breast_cancer.data[400:, np.newaxis, 3] # Also use 4th feature
y_test = breast_cancer.target[400:, np.newaxis]

# Normalize X_train and X_test
normalize = lambda matrix: (matrix - np.min(matrix, axis=0)) / (np.max(matrix, axis=0) - np.min(matrix, axis=0)) # Lambda expression
X_train_norm = normalize(X_train)
X_test_norm = normalize(X_test)

# Create several functions to do everything in the prediction phase
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
    predictions = np.asarray([knn_predict(entry, X_train, y_train, k, kind=kind) for entry in X_test]).reshape(y_test.shape)
    return predictions

# Get data from breast cancer dataset
X_train = breast_cancer.data[:400, np.newaxis, 3]
y_train = breast_cancer.target[:400, np.newaxis]
X_test = breast_cancer.data[400:, np.newaxis, 3]
y_test = breast_cancer.target[400:, np.newaxis]

# Normalize X_train and X_test
X_train_norm = normalize(X_train)
X_test_norm = normalize(X_test)

# Try for k = 5
k = 5
y_pred = nearest_neighbors(X_test_norm, X_train_norm, y_train, k)

# accuracy = lambda y_test, y_pred: 
accuracy = lambda y_test, y_pred: len(np.where(y_pred.flatten() == y_test.flatten())[0]) / len(y_test)

print(f"Accuracy exercise 3: {accuracy(y_test, y_pred)}")

#### Exercise 4: k-NN regression ####
# Get data from diabetes dataset
# X_train = diabetes.data[:300, np.newaxis, 3]
# y_train = diabetes.target[:300, np.newaxis]
# X_test = diabetes.data[300:, np.newaxis, 3]
# y_test = diabetes.target[300:, np.newaxis]
X_train = diabetes.data[:300, :]
y_train = diabetes.target[:300, np.newaxis]
X_test = diabetes.data[300:, :]
y_test = diabetes.target[300:, np.newaxis]

# Compute and print mean squared error
y_pred = nearest_neighbors(X_test, X_train, y_train, 5, "regression")
print(f"Mean squared error exercise 4: {mean_squared_error(y_test, y_pred)}")

# Do it for different values of k
ks = np.arange(1, 26)
mses = np.zeros_like(ks)

for i, k in enumerate(ks):
    y_pred_k = nearest_neighbors(X_test, X_train, y_train, k, "regression")
    mse_k = mean_squared_error(y_test, y_pred_k)
    mses[i] = mse_k

# Plot the mean squared errors
plt.figure()
plt.plot(ks, mses, marker=".", label="KNN Regression", color="blue")
plt.axhline(mse_ex1, label="Linear Regression", color="orange")
plt.legend(loc="best")
plt.xlabel("k")
plt.ylabel("MSE")
plt.title("MSE for Different K-Values (Diabetes Dataset)")
plt.show()

#### Exercise 5: Class-conditional probability
# Get data from breast cancer dataset
X = breast_cancer.data
y = breast_cancer.target

# Split X into 0 and 1
X_0 = X[y == 0]
X_1 = X[y == 1]

# Get means and std's
X_0_means = np.mean(X_0, axis=0)
X_0_stds = X_0.std(axis=0)
X_1_means = np.mean(X_1, axis=0)
X_1_stds = X_1.std(axis=0)

# Store feature names in array
features = breast_cancer.feature_names

# Plot over all features
fig, axes = plt.subplots(nrows=6, ncols=5, figsize=(24, 8))

for i, ax in enumerate(axes.flatten()):
    mean_0, sd_0 = X_0_means[i], X_0_stds[i]
    mean_1, sd_1 = X_1_means[i], X_1_stds[i]
    feature = features[i]

    # Generate x-axis values symmetrically around the mean
    factor = 4
    x_axis_0 = np.arange(mean_0 - factor * sd_0, mean_0 + factor * sd_0, 0.001)
    x_axis_1 = np.arange(mean_1 - factor * sd_1, mean_1 + factor * sd_1, 0.001)

    # Plot the Gaussian distributions
    ax.plot(x_axis_0, norm.pdf(x_axis_0, mean_0, sd_0), label="0")
    ax.plot(x_axis_1, norm.pdf(x_axis_1, mean_1, sd_1), label="1")
    ax.fill_between(x_axis_0, norm.pdf(x_axis_0, mean_0, sd_0)) # fill_between: obtained by using ChatGPT
    ax.fill_between(x_axis_1, norm.pdf(x_axis_1, mean_1, sd_1), alpha=0.4)
    
    # Set feature name as title
    ax.set_title(feature, fontsize=10)

    # Set ticks to smaller fontsize
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=7)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=7)

# Create custom legend handles -- obtained by asking ChatGPT
legend_handles = [
    Line2D([0], [0], color='blue', lw=2, label='malignant'),
    Line2D([0], [0], color='orange', lw=2, label='benign')
]

# Add a single legend for the entire figure
fig.legend(handles=legend_handles, loc=[0.9, 0.92], fontsize=8)

fig.suptitle('Gaussian Distributions for All Features', fontsize=15)

# Adjust the space between subplots
plt.subplots_adjust(wspace=0.3, hspace=1.0) 

# Save and close figure
if not(os.path.exists(r"class_conditional_probabilities.png")):
    plt.savefig(r"class_conditional_probabilities.png", bbox_inches="tight")
plt.show()