"""
Synthetic data generator for benchmarking Bayesian neural networks.

Generates inputs from a standard normal distribution and targets as a nonlinear
additive function of the first four features, plus Gaussian noise. Supports train-test splitting.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons
from ucimlrepo import fetch_ucirepo


def generate_Friedman_data(N=200, D=10, sigma=1.0, test_size=0.2, seed=42, standardize_y=True):
    """
    Generate synthetic regression data for Bayesian neural network experiments.

    Parameters:
        N (int): Number of samples.
        D (int): Number of features.
        sigma (float): Noise level.
        test_size (float): Proportion for test split.
        seed (int): Random seed.
        standardize_y (bool): Whether to standardize the response variable.

    Returns:
        tuple: (X_train, X_test, y_train, y_test, y_mean, y_std) if standardize_y,
               else (X_train, X_test, y_train, y_test)
    """
    np.random.seed(seed)
    X = np.random.uniform(0, 1, size=(N, D))
    x0, x1, x2, x3, x4 = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4]

    y_clean = (
        10 * np.sin(np.pi * x0 * x1) +
        20 * (x2 - 0.5) ** 2 +
        10 * x3 +
        5.0 * x4
    )

    y = y_clean + np.random.normal(0, sigma, size=N)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

    if standardize_y:
        y_mean = y_train.mean()
        y_std = y_train.std() if y_train.std() > 0 else 1.0  # avoid division by zero

        y_train = (y_train - y_mean) / y_std
        y_test = (y_test - y_mean) / y_std

        return X_train, X_test, y_train, y_test

    return X_train, X_test, y_train, y_test

def generate_GAM_data(N=500, D=5, sigma=0.5, test_size=0.2, config=1, seed=42):
    """
    Generate synthetic regression data for Bayesian neural network experiments.

    The target variable is a nonlinear function of the first four input features:
        y = 2.0 * sin(x0) - 1.5 * tanh(x1) + 0.7 * x2^2 - 0.5 * exp(-x3^2) + noise

    Parameters:
        N (int): Total number of samples to generate. Default is 500.
        D (int): Number of input features. Default is 5.
        sigma (float): Standard deviation of the additive Gaussian noise. Default is 0.5.
        test_size (float): Proportion of data to allocate to the test set. Default is 0.2.
        seed (int): Random seed for reproducibility. Default is 42.

    Returns:
        tuple: (X_train, X_test, y_train, y_test), where each is a NumPy array.
    """

    np.random.seed(seed)
    X = np.random.normal(0, 1, size=(N, D))
    x0, x1, x2, x3, x4 = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4]

    y_clean_1 = (
        2.0 * np.sin(np.pi * x0) +          # smooth periodic effect
        3.0 * (x1 - 0.5)**2 +                     # quadratic bump
        np.exp(x2) +                        # exponential growth
        2.0 * x3 +                          # linear term
        np.log1p(np.abs(x4))                # smooth logarithmic transform
    )
    
    y_clean_2 = (
        3.0 * np.cos(2 * np.pi * x0) +
        2.0 * x1 * np.sin(np.pi * x1) +
        1.5 * np.tanh(3 * x2 - 1.5) +
        0.5 * x3**3 +
        2.0 * np.sqrt(np.abs(x4))
    )
    
    y_clean_3 = (
        4.0 * (x0 > 0.5).astype(float) +
        3.0 * np.maximum(0, x1 - 0.3)**1.5 +
        -1.0 * np.exp(-5 * np.abs(x2)) +
        2.5 * np.sin(np.pi * x3) * np.exp(-x3) +
        1.5 * np.arctan(5 * (x4 - 0.5))
    )
    
    if config == 1:
        y_clean = y_clean_1
    elif config == 2:
        y_clean = y_clean_2
    elif config == 3:
        y_clean = y_clean_3
    else:
        raise ValueError("Invalid configuration. Use 1, 2 or 3.")
    
    y = y_clean + np.random.normal(0, sigma, size=N)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

    return X_train, X_test, y_train, y_test

def generate_interaction_data(N=500, D=5, sigma=0.5, test_size=0.2, seed=42):
    """
    Generate synthetic regression data for Bayesian neural network experiments.

    The target variable is a nonlinear function of the first four input features:
        y = 2.0 * sin(x0) - 1.5 * tanh(x1) + 0.7 * x2^2 - 0.5 * exp(-x3^2) + noise

    Parameters:
        N (int): Total number of samples to generate. Default is 500.
        D (int): Number of input features. Default is 5.
        sigma (float): Standard deviation of the additive Gaussian noise. Default is 0.5.
        test_size (float): Proportion of data to allocate to the test set. Default is 0.2.
        seed (int): Random seed for reproducibility. Default is 42.

    Returns:
        tuple: (X_train, X_test, y_train, y_test), where each is a NumPy array.
    """

    np.random.seed(seed)
    X = np.random.normal(0, 1, size=(N, D))
    x0, x1, x2, x3, x4 = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4]

    y_interactions = (
    np.sin(np.pi * (x0 + x1 * x2)) +
    3.0 * x2 * x3 +
    np.exp(-x4 * x0) +
    0.7 * np.tanh(x1 * x4 + x0 * x2)
    )

    y = y_interactions + np.random.normal(0, sigma, size=N)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

    return X_train, X_test, y_train, y_test

def generate_two_moons(N=200, sigma=0.2, test_size=0.2, D=2, seed=42):
    """
    Generate a two moons dataset with optional redundant features and split into training and test sets.

    Parameters:
    n_samples (int): Number of samples to generate.
    noise (float): Standard deviation of Gaussian noise added to the data.
    test_size (float): Proportion of the dataset to include in the test split.
    n_features (int): Total number of features (must be >=2). Extra features will be random noise.
    random_state (int): Random seed for reproducibility.

    Returns:
    X_train (ndarray): Training features.
    X_test (ndarray): Test features.
    y_train (ndarray): Training labels.
    y_test (ndarray): Test labels.
    """
    if D < 2:
        raise ValueError("n_features must be at least 2.")

    X, y = make_moons(n_samples=N, noise=sigma, random_state=seed)
    
    y = np.where(y == 0, 1, 2)  

    if D > 2:
        rng = np.random.default_rng(seed=seed)
        extra_features = rng.normal(0, 1, size=(N, D - 2))
        X = np.hstack((X, extra_features))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
    return X_train, X_test, y_train, y_test

def load_abalone_regression_data(
    path="datasets/abalone/abalone.csv",
    target="Rings",
    frac=0.5,
    standardized = False,
    random_state=42,
    test_size=0.2
):
    """
    Load and preprocess the abalone dataset for regression.

    Parameters:
        path (str): Path to the abalone CSV file.
        frac (float): Fraction of data to sample.
        random_state (int): Random seed.
        test_size (float): Proportion of data to use for testing.

    Returns:
        X_train, X_test, y_train, y_test
    """
    column_names = [
        "Sex", "Length", "Diameter", "Height",
        "Whole weight", "Shucked weight", "Viscera weight",
        "Shell weight", "Rings"
    ]

    abalone = pd.read_csv(path, header=None, names=column_names)

    # Map sex to integers: I=1, F=2, M=3
    abalone['Sex'] = abalone['Sex'].map({'I': 1, 'F': 2, 'M': 3})
    abalone = abalone.sample(frac=frac, random_state=random_state).reset_index(drop=True)
    
    X = abalone.drop([target], axis=1)
    numeric_cols = X.select_dtypes(include='number').columns.drop('Sex')
    scaler = StandardScaler()
    X_scaled = X.copy()
    X_scaled[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    y_raw = abalone[target].astype(int)
    y = y_raw.values.astype(float)
    
    if standardized:
        return train_test_split(X_scaled, y, test_size=test_size, random_state=random_state)
    else:
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

def load_boston_regression_data(
    path="UCI_data/boston.csv",
    target="MEDV",
    frac=0.5,
    standardized = False,
    random_state=42,
    test_size=0.2
):
    """
    Load and preprocess the abalone dataset for regression.

    Parameters:
        path (str): Path to the abalone CSV file.
        frac (float): Fraction of data to sample.
        random_state (int): Random seed.
        test_size (float): Proportion of data to use for testing.

    Returns:
        X_train, X_test, y_train, y_test
    """
    column_names = [
        "CRIM", "ZN", "INDUS", "CHAS",
        "NOX", "RM", "AGE",
        "DIS", "RAD", "TAX", "PTRATIO",
        "B", "LSTAT", "MEDV"
    ]

    boston = pd.read_csv(path, header=None, names=column_names)
    if boston[target].iloc[0] == target:
        boston = boston.drop(index=0).reset_index(drop=True)

    # Map sex to integers: I=1, F=2, M=3
    if frac != 1.0:
        boston = boston.sample(frac=frac, random_state=random_state).reset_index(drop=True)
    
    #X = boston.drop([target], axis=1)
    X = boston.drop([target], axis=1)
    X = X.astype(float)  # force all features to float

    y = boston[target].astype(float).to_numpy()#.astype(float)

    
    if standardized:
        x_scaler = StandardScaler()
        y_scaler = StandardScaler()

        X_scaled = x_scaler.fit_transform(X)
        y_scaled = y_scaler.fit_transform(y.reshape(-1, 1)).flatten()

        return train_test_split(X_scaled, y_scaled, test_size=test_size, random_state=random_state)
    else:
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

def load_concrete_regression_data(
    path="UCI_data/concrete.csv",
    target="MPa",
    frac=0.5,
    standardized = False,
    random_state=42,
    test_size=0.2
):
    """
    Load and preprocess the abalone dataset for regression.

    Parameters:
        path (str): Path to the abalone CSV file.
        frac (float): Fraction of data to sample.
        random_state (int): Random seed.
        test_size (float): Proportion of data to use for testing.

    Returns:
        X_train, X_test, y_train, y_test
    """
    column_names = [
        "Cement", "BFS", "FlyAsh", "Water",
        "Superplasticizer", "CoarseAggregate", "FineAggregate",
        "Age", "MPa"
    ]

    concrete = pd.read_csv(path, header=None, names=column_names)
    if concrete[target].iloc[0] == target:
        concrete = concrete.drop(index=0).reset_index(drop=True)

    # Map sex to integers: I=1, F=2, M=3
    if frac != 1.0:
        concrete = concrete.sample(frac=frac, random_state=random_state).reset_index(drop=True)
    
    #X = boston.drop([target], axis=1)
    X = concrete.drop([target], axis=1)
    X = X.astype(float)  # force all features to float

    y = concrete[target].astype(float).to_numpy()#.astype(float)

    
    if standardized:
        x_scaler = StandardScaler()
        y_scaler = StandardScaler()

        X_scaled = x_scaler.fit_transform(X)
        y_scaled = y_scaler.fit_transform(y.reshape(-1, 1)).flatten()

        return train_test_split(X_scaled, y_scaled, test_size=test_size, random_state=random_state)
    else:
        return train_test_split(X, y, test_size=test_size, random_state=random_state)
    
def load_breast_cancer_data(test_size=0.2, standardize=True, random_state=42):
    """
    Load the UCI Breast Cancer Wisconsin (Diagnostic) dataset, split into train/test, optionally standardize.

    Parameters:
        test_size (float): Proportion of the data to include in the test split.
        standardize (bool): Whether to standardize the feature matrix X.
        random_state (int): Random seed for reproducibility.

    Returns:
        X_train, X_test, y_train, y_test, (mean_, scale_) if standardize else (None, None)
    """
    # Load data from local file or download manually first:
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
    columns = ['ID', 'Diagnosis'] + [f'feature_{i}' for i in range(1, 31)]
    
    data = pd.read_csv(url, header=None, names=columns)
    
    # Map target labels to binary
    data['Diagnosis'] = data['Diagnosis'].map({'M': 2, 'B': 1})

    X = data.drop(['ID', 'Diagnosis'], axis=1).values
    y = data['Diagnosis'].values

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    mean_, scale_ = None, None
    if standardize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        mean_, scale_ = scaler.mean_, scaler.scale_

    return X_train, X_test, y_train, y_test, mean_, scale_

def load_wine_quality_data(test_size=0.2, standardize=True, random_state=42):
    """
    Load the UCI Wine Quality dataset (white and red combined), split into train/test, and optionally standardize.

    Parameters:
        test_size (float): Proportion of data for test split.
        standardize (bool): Whether to standardize the feature matrix X.
        random_state (int): Random seed for reproducibility.

    Returns:
        X_train, X_test, y_train, y_test, mean_, scale_
    """
    # Fetch the dataset
    wine_quality = fetch_ucirepo(id=186)
    
    # Extract features and target
    X = wine_quality.data.features
    y = wine_quality.data.targets["quality"]

    # Convert to NumPy arrays
    X = X.to_numpy()
    y = y.to_numpy()

    # Optional: shift quality labels to be 1-based (optional, depends on your Stan model)
    # min_quality = y.min()  # Often 3
    # y = y - min_quality + 1

    # Split into train/test with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    mean_, scale_ = None, None
    if standardize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        mean_, scale_ = scaler.mean_, scaler.scale_

    return X_train, X_test, y_train, y_test, mean_, scale_

def generate_uniform_data(N=200, D=10, test_size=0.2, seed=42,
                          output_nodes=1, y_mode="zeros", standardize_y=False):
    """
    Generate X ~ Uniform(-1, 1) for prior-only runs.
    Returns train/test splits and dummy y (unused by prior-only Stan).

    Parameters:
        N (int): number of samples
        D (int): number of features
        test_size (float): fraction for test split (0..1)
        seed (int): RNG seed
        output_nodes (int): dimension of y (usually 1)
        y_mode (str): 'zeros' | 'noise' | 'sum' (dummy targets; 'zeros' is fine)
        standardize_y (bool): kept for API parity; ignored unless y_mode != 'zeros'

    Returns:
        (X_train, X_test, y_train, y_test)
        If output_nodes == 1, y arrays are 1D; else shape is (N, output_nodes).
    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, size=(N, D))

    if y_mode == "zeros":
        y = np.zeros((N, output_nodes))
    elif y_mode == "noise":
        y = rng.normal(0.0, 1e-6, size=(N, output_nodes))  # tiny variance to avoid 0-std edge cases
    elif y_mode == "sum":
        y = X.sum(axis=1, keepdims=True).repeat(output_nodes, axis=1)
    else:
        raise ValueError("y_mode must be one of: 'zeros', 'noise', 'sum'")

    # manual train/test split
    n_test = int(round(test_size * N))
    idx = rng.permutation(N)
    test_idx, train_idx = idx[:n_test], idx[n_test:]

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # match your previous API: 1D y when output_nodes == 1
    if output_nodes == 1:
        y_train = y_train.ravel()
        y_test = y_test.ravel()

    # optional y standardization (won't matter for prior-only; kept for compatibility)
    if standardize_y and output_nodes == 1 and y_mode != "zeros":
        y_mean = y_train.mean()
        y_std = y_train.std() if y_train.std() > 0 else 1.0
        y_train = (y_train - y_mean) / y_std
        y_test = (y_test - y_mean) / y_std

    return X_train, X_test, y_train, y_test

def generate_latent_hastie_data(n, p, 
                                d=5, r_theta=1.0, sigma_xi=0.0, rng=None,
                                random_state=42,
                                test_size=0.2):
    """
    Section 5.4 latent model (Hastie–Montanari–Rosset–Tibshirani):
      X = Z W^T + U,   y = Z θ + ξ
      z_i ~ N(0, I_d), u_ij ~ N(0, 1), ξ_i ~ N(0, σ_ξ^2)
    Rows w_j of W satisfy ||w_j|| = 1.               [Fig. 5/6 setup]
    Population mapping to linear model:
      Σ = I_p + W W^T,   β = W (I + W^T W)^{-1} θ.   [eqs. (26)-(27)]
    Returns: X (n×p), y (n,), W (p×d), theta (d,), beta_true (p,), Sigma (p×p)
    """

    rng = np.random.default_rng(random_state)

    # Random W with unit-norm rows (||w_j||=1)
    W = rng.normal(size=(p, d))
    W /= np.linalg.norm(W, axis=1, keepdims=True) + 1e-12  # enforce ||w_j||=1

    # Latent Z, feature noise U, label noise ξ
    Z = rng.normal(size=(n, d))
    U = rng.normal(size=(n, p))
    xi = rng.normal(scale=sigma_xi, size=n)

    # Signal vector θ with ||θ|| = r_theta
    theta = rng.normal(size=d)
    theta *= r_theta / (np.linalg.norm(theta) + 1e-12)

    # Data
    X = Z @ W.T + U
    y = Z @ theta + xi

    # Population quantities for risk
    #Sigma = np.eye(p) + W @ W.T
    #beta_true = W @ np.linalg.solve(np.eye(d) + W.T @ W, theta)  # β = W (I + W^T W)^(-1) θ

    return train_test_split(X, y, test_size=test_size, random_state=random_state)

