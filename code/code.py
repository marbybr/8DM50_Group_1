import numpy as np
from collections import Counter
from scipy.stats import mode
import matplotlib.pyplot as plt


def lsq(X, y):
    """
    Least squares linear regression
    :param X: Input data matrix
    :param y: Target vector
    :return: Estimated coefficient vector for the linear regression
    """

    # add column of ones for the intercept
    ones = np.ones((len(X), 1))
    X = np.concatenate((ones, X), axis=1)

    # calculate the coefficients
    beta = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, y))

    return beta

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

# define KNN classifier
def KNN_classifier(X_train, y_train, X_test, k):
    # Find shortest distances between points, in order to identify nearest neighbor with (x-y)^2 = x^2 + y^2 - 2xy

    #reschape matrices for following operations for getting x^2 and y^2 in formula above
    X_train_squared = np.sum(X_train**2, axis=1).reshape(-1, 1)
    X_test_squared = np.sum(X_test**2, axis=1).reshape(1, -1)
    
    # calculate -2xy and complete formula and transpose nearness matrix
    nearness = X_train_squared + X_test_squared - 2 * np.dot(X_train, X_test.T)
    nearness = nearness.T

    # Sort nearness and get indices of the k lowest distances
    nearest = np.argsort(nearness, axis=1)[:, :k]

    # find the nearest neighbors
    nearest_neighbors = y_train[nearest]

    # find most common feature of neigborhood
    predictions, _ = mode(nearest_neighbors, axis=1)
    #flatten predictions to a one dimensional array
    predictions.flatten()
    
    return predictions

def performance_test_ACC(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)

def performance_tester_ACC(X_train, y_train, X_test, y_test, k_range):
    k_list = list(k_range)
    performances = []
    for k in k_list:
        y_pred = KNN_classifier(X_train, y_train, X_test, k)
        performance = performance_test_ACC(y_test, y_pred)
        performances.append(performance)
    
    # Make a graph showing performance
    plt.figure(figsize=(10, 6))
    plt.plot(k_list, performances, marker='o', linestyle='-')
    plt.title('k-NN Classifier perormance for a range of k Values')
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Performance(accuracy)')
    plt.grid(True)
    plt.show()

# define KNN linear regressor
def KNN_linear_regressor(X_train, y_train, X_test, k):
    # Find shortest distances between points, in order to identify nearest neighbor with (x-y)^2 = x^2 + y^2 - 2xy

    #reschape matrices for following operations for getting x^2 and y^2 in formula above
    X_train_squared = np.sum(X_train**2, axis=1).reshape(-1, 1)
    X_test_squared = np.sum(X_test**2, axis=1).reshape(1, -1)
    
    # calculate -2xy and complete formula and transpose nearness matrix
    nearness = X_train_squared + X_test_squared - 2 * np.dot(X_train, X_test.T)
    nearness = nearness.T
    # Sort nearness and get indices of the k lowest distances
    nearest = np.argsort(nearness, axis=1)[:, :k]

    # find the nearest neighbors
    nearest_neighbors = y_train[nearest]

    # find most common feature of neigborhood
    predictions, _ = mode(nearest_neighbors, axis=1)
    #flatten predictions to a one dimensional array
    predictions.flatten()
    
    return predictions   

#define a measure for the performance of the knn classifier, here the mean square error
def performance_test_MSE(y_true, y_pred):
    return np.mean((y_pred - y_true)**2)

def performance_tester_MSE(X_train, y_train, X_test, y_test, k_range):
    k_list = list(k_range)
    performances = []
    for k in k_list:
        y_pred = KNN_linear_regressor(X_train, y_train, X_test, k)
        performance = performance_test_MSE(y_test, y_pred)
        performances.append(performance)
    
    # Make a graph showing performance
    plt.figure(figsize=(10, 6))
    plt.plot(k_list, performances, marker='o', linestyle='-')
    plt.title('k-NN Linear Regressor perormance for a range of k Values')
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Performance(MSE)')
    plt.grid(True)
    plt.show()
    
    
# Function to calculate the Gaussian probability density function
def gaussian_pdf(x, mean, variance):
    return (1.0 / np.sqrt(2 * np.pi * variance)) * np.exp(-((x - mean) ** 2) / (2 * variance))

# Calculate the class-conditional probability for all features
def class_conditional_probabilities(X, mean_class_0, var_class_0, mean_class_1, var_class_1, feature_names):
    probs_class_0 = {}
    probs_class_1 = {}
    for feature in feature_names:
        probs_class_0[feature] = gaussian_pdf(X[feature], mean_class_0[feature], var_class_0[feature])
        probs_class_1[feature] = gaussian_pdf(X[feature], mean_class_1[feature], var_class_1[feature])
    return probs_class_0, probs_class_1


