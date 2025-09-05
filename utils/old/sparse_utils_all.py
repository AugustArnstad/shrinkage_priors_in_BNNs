import numpy as np
from scipy.special import softmax
from sklearn.metrics import mean_squared_error
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import re
import os

def relu(x):
    return np.maximum(0, x)

def prune_weights(W, num_to_prune):
    """
    Create a binary mask that keeps the largest (by absolute value) weights.
    """
    flat = np.abs(W.flatten())
    if num_to_prune == 0:
        return np.ones_like(W)
    idx = np.argpartition(flat, num_to_prune)[:num_to_prune]
    mask_flat = np.ones_like(flat, dtype=bool)
    mask_flat[idx] = False
    return mask_flat.reshape(W.shape).astype(float)

def forward_pass_precise(X, W1, b1, W2, b2, task):
    pre_hidden = X @ W1 + b1.reshape(1, -1)
    hidden = relu(pre_hidden)
    output = hidden @ W2 + b2.reshape(1, -1)

    if task == "classification":
        return softmax(output, axis=1)
    elif task == "regression":
        return output
    else:
        raise ValueError(f"Unknown task: {task}")

def evaluate_performance(preds, y_true, task):
    if task == "classification":
        y_pred = np.argmax(preds, axis=1)
        return np.mean(y_pred == y_true)
    elif task == "regression":
        return np.sqrt(np.mean((preds - y_true) ** 2))  # RMSE

def extract_posterior_samples(fit):
    """
    Extract and format posterior samples from a fitted Stan model.
    """
    W1_samples = np.transpose(fit.stan_variable("data_to_hidden"), (0, 2, 1))  # (S, P, H)
    W2_samples = fit.stan_variable('hidden_to_output')                         # (S, H, K)
    b1_samples = fit.stan_variable('hidden_biases')                            # (S, L, H)
    b2_samples = fit.stan_variable('output_bias')                              # (S, K)
    return W1_samples, W2_samples, b1_samples, b2_samples

def predict_with_pruning(W1_samples, W2_samples, b1_samples, b2_samples, X_test, y_test, sparsity_level, task):
    """
    Perform sample-wise pruning and prediction, return accuracy and mean pruned weights.
    """
    S, P, H = W1_samples.shape
    total_weights = P * H
    num_to_prune = int(np.floor(sparsity_level * total_weights))

    means_samples = []
    mean_W1 = np.zeros((P, H))  # for visualization

    for s in range(S):
        W1 = W1_samples[s]
        W2 = W2_samples[s]
        b1 = b1_samples[s][0]
        b2 = b2_samples[s]

        mask = prune_weights(W1, num_to_prune)
        W1_pruned = W1 * mask
        mean_W1 += W1_pruned

        means = forward_pass_precise(X_test, W1_pruned, b1, W2, b2, task)
        means_samples.append(means)

    means_avg = np.mean(means_samples, axis=0)
    metric =np.sqrt(mean_squared_error(y_test, means_avg))
    return metric 

def compute_pruned_rmse_over_sparsity(
    all_fits,
    model_prior,
    base_path='stored_datasets/regression',
    sparsity_levels=None,
    task='regression',
    verbose=False
):
    """
    Evaluate RMSE over different sparsity levels for all models using the given prior.

    Parameters
    ----------
    all_fits : dict
        Dictionary mapping model keys (e.g., 'GAM_N100_p8_sigma1.00_seed1') to fit results.
    model_prior : str
        The name of the prior model to extract (e.g., 'Dirichlet Horseshoe').
    base_path : str
        Path to the directory containing `.npz` test datasets.
    sparsity_levels : list of float
        Sparsity levels to evaluate. Default: [0.0, ..., 0.95].
    task : str
        'regression' or 'classification'
    verbose : bool
        Whether to print progress.

    Returns
    -------
    all_rmse_results : dict
        Dictionary mapping model_key to list of RMSE values for each sparsity level.
    """

    if sparsity_levels is None:
        sparsity_levels = [0.0, 0.25, 0.50, 0.60, 0.7, 0.8, 0.85, 0.90, 0.95, 0.99]

    all_rmse_results = {}

    for model_key in all_fits:
        if model_prior not in all_fits[model_key]:
            if verbose:
                print(f"Model '{model_prior}' not found in {model_key}, skipping.")
            continue

        data_path = os.path.join(base_path, f"{model_key}.npz")
        if not os.path.exists(data_path):
            if verbose:
                print(f"Missing data file for {model_key}, skipping.")
            continue

        if verbose:
            print(f"Processing {model_key}...")

        data = np.load(data_path)
        X_test = data['X_test']
        y_test = data['y_test']

        fit = all_fits[model_key][model_prior]['posterior']
        W1_samples, W2_samples, b1_samples, b2_samples = extract_posterior_samples(fit)

        rmse_list = []
        for sparsity in sparsity_levels:
            rmse = predict_with_pruning(
                W1_samples, W2_samples, b1_samples, b2_samples,
                X_test, y_test, sparsity, task
            )
            rmse_list.append(rmse)

        all_rmse_results[model_key] = rmse_list

    return all_rmse_results

def plot_rmse_degradation_grid(
    results_by_prior,
    sparsity_levels,
    title="RMSE Degradation by Sparsity Level",
    figsize=(18, 5),
):
    """
    Plot RMSE degradation over sparsity levels, grouped by sigma and colored by N.
    Each row corresponds to one prior (i.e., one model type).
    
    Parameters
    ----------
    results_by_prior : dict
        Dictionary mapping model_prior (e.g. 'Dirichlet Horseshoe') to all_rmse_results dict.
    sparsity_levels : list of float
        Sparsity levels used in evaluation.
    title : str
        Title for the entire figure.
    figsize : tuple
        Base figure size. Adjusts based on number of priors.
    """
    
    def parse_sigma(model_key):
        match = re.search(r'sigma([\d.]+)', model_key)
        return float(match.group(1)) if match else None

    def parse_N(model_key):
        match = re.search(r'GAM_N(\d+)', model_key)
        return int(match.group(1)) if match else None

    priors = list(results_by_prior.keys())
    sigmas = [1.0, 3.0, 5.0]
    n_rows = len(priors)
    n_cols = len(sigmas)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0], figsize[1] * n_rows), sharey=False)

    if n_rows == 1:
        axes = np.expand_dims(axes, 0)  # ensure 2D array

    for row_idx, prior in enumerate(priors):
        all_rmse_results = results_by_prior[prior]
        Ns = sorted(set(parse_N(k) for k in all_rmse_results))
        color_map = {N: cm.viridis(i / (len(Ns)-1)) for i, N in enumerate(Ns)}

        for col_idx, sigma in enumerate(sigmas):
            ax = axes[row_idx, col_idx]

            for model_key, rmse_list in all_rmse_results.items():
                if parse_sigma(model_key) == sigma:
                    N = parse_N(model_key)
                    ax.plot(
                        sparsity_levels,
                        rmse_list,
                        marker='o',
                        color=color_map[N],
                        label=f'N={N}'
                    )

            ax.set_title(f"σ = {sigma}")
            ax.set_xlabel("Sparsity Level")
            if col_idx == 0:
                ax.set_ylabel(f"{prior}\n\nRMSE")
            else:
                ax.set_ylabel("")

            ax.grid(True)

            # One legend per subplot, deduplicated by label
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), title="N", fontsize='small')

    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

