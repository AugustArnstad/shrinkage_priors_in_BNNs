import numpy as np
import pandas as pd


def forward_pass_relu(X, W1, b1, W2, b2):
    """
    Forward pass for a single layer BNN.
    """
    pre_act_1 = X @ W1 + b1.reshape(1, -1)
    #pre_hidden += b1.reshape(1, -1)
    post_act_1 = np.maximum(0, pre_act_1)
    ouput = post_act_1 @ W2 + b2.reshape(1, -1)
    return ouput

def forward_pass_tanh(X, W1, b1, W2, b2):
    """
    Forward pass for a single layer BNN.
    """
    pre_act_1 = X @ W1 + b1.reshape(1, -1)
    #pre_hidden += b1.reshape(1, -1)
    post_act_1 = np.tanh(pre_act_1)
    ouput = post_act_1 @ W2 + b2.reshape(1, -1)
    return ouput

def compute_sparse_rmse_results(seeds, models, all_fits, get_N_sigma, forward_pass,
                         sparsity=0.0, prune_fn=None):
    results = []
    posterior_means = []

    for seed in seeds:
        N, sigma = get_N_sigma(seed)
        dataset_key = f'Friedman_N{N}_p10_sigma{sigma:.2f}_seed{seed}'
        path = f"datasets/friedman/{dataset_key}.npz"

        try:
            data = np.load(path)
            X_test, y_test = data["X_test"], data["y_test"]
        except FileNotFoundError:
            print(f"[SKIP] File not found: {path}")
            continue

        for model in models:
            try:
                fit = all_fits[dataset_key][model]['posterior']
                W1_samples = fit.stan_variable("W_1")           # (S, P, H)
                W2_samples = fit.stan_variable("W_L")           # (S, H, O)
                b1_samples = fit.stan_variable("hidden_bias")   # (S, O, H)
                b2_samples = fit.stan_variable("output_bias")   # (S, O)
            except KeyError:
                print(f"[SKIP] Model or posterior not found: {dataset_key} -> {model}")
                continue

            S = W1_samples.shape[0]
            rmses = np.zeros(S)
            #print(y_test.shape)
            y_hats = np.zeros((S, y_test.shape[0]))

            for i in range(S):
                W1 = W1_samples[i]
                W2 = W2_samples[i]

                # Apply pruning mask if requested
                if prune_fn is not None and sparsity > 0.0:
                    masks = prune_fn([W1, W2], sparsity)
                    W1 = W1 * masks[0]
                    #W2 = W2 * masks[1]

                y_hat = forward_pass(X_test, W1, b1_samples[i][0], W2, b2_samples[i])
                y_hats[i] = y_hat.squeeze()  # Store the prediction for each sample
                rmses[i] = np.sqrt(np.mean((y_hat.squeeze() - y_test)**2))
                
            posterior_mean = np.mean(y_hats, axis=0)
            posterior_mean_rmse = np.sqrt(np.mean((posterior_mean - y_test.squeeze())**2))

            posterior_means.append({
                'seed': seed,
                'N': N,
                'sigma': sigma,
                'model': model,
                'sparsity': sparsity,
                'posterior_mean_rmse': posterior_mean_rmse
            })

            for i in range(S):
                results.append({
                    'seed': seed,
                    'N': N,
                    'sigma': sigma,
                    'model': model,
                    'sparsity': sparsity,
                    'rmse': rmses[i]
                })

    df_rmse = pd.DataFrame(results)
    df_posterior_rmse = pd.DataFrame(posterior_means)

    return df_rmse, df_posterior_rmse

def global_prune_weights(weight_matrices, sparsity_level):
    """
    Prune globally across multiple weight matrices.
    
    Args:
        weight_matrices: List of numpy arrays (weight matrices).
        sparsity_level: Float in [0, 1], fraction of weights to prune.

    Returns:
        List of binary masks with same shapes as weight_matrices.
    """
    # Flatten all weights and concatenate
    flat_weights = np.concatenate([w.flatten() for w in weight_matrices])
    abs_weights = np.abs(flat_weights)
    
    # Determine number of weights to prune
    total_weights = abs_weights.size
    num_to_prune = int(np.floor(sparsity_level * total_weights))

    # Get indices of smallest weights to prune
    prune_indices = np.argpartition(abs_weights, num_to_prune)[:num_to_prune]
    
    # Create global mask
    global_mask_flat = np.ones(total_weights, dtype=bool)
    global_mask_flat[prune_indices] = False

    # Split the global mask back into original shapes
    masks = []
    idx = 0
    for w in weight_matrices:
        size = w.size
        mask = global_mask_flat[idx:idx + size].reshape(w.shape)
        masks.append(mask.astype(float))
        idx += size

    return masks

def local_prune_weights(weights, sparsity_level, index_to_prune=0):
    """
    Apply pruning to only one weight matrix in a list, specified by index.

    Parameters:
    - weights: list of np.ndarray (e.g., [W1, W2])
    - sparsity_level: fraction of weights to prune (0.0 to 1.0)
    - index_to_prune: which weight matrix to prune in the list

    Returns:
    - list of masks (one for each weight matrix)
    """
    masks = [np.ones_like(W) for W in weights]

    W = weights[index_to_prune]
    flat = np.abs(W.flatten())
    num_to_prune = int(np.floor(sparsity_level * flat.size))

    if num_to_prune > 0:
        idx = np.argpartition(flat, num_to_prune)[:num_to_prune]
        mask_flat = np.ones_like(flat, dtype=bool)
        mask_flat[idx] = False
        masks[index_to_prune] = mask_flat.reshape(W.shape).astype(float)

    return masks

def prune_nodes_by_output_weights(weight_matrices, sparsity_level, index_to_prune=1):
    """
    Prune entire nodes based on the outgoing weights from a chosen weight matrix.

    Parameters:
    - weight_matrices: list of np.ndarray (e.g., [W1, W2])
    - sparsity_level: fraction of nodes to prune (0.0 to 1.0)
    - index_to_prune: which weight matrix's outgoing connections to use (default: second layer)

    Returns:
    - list of masks (one for each weight matrix)
    """
    masks = [np.ones_like(W) for W in weight_matrices]

    W = weight_matrices[index_to_prune]  # Shape: (H, O)
    node_norms = np.linalg.norm(W, axis=1)  # L2 norm per hidden node

    num_nodes = W.shape[0]
    num_to_prune = int(np.floor(sparsity_level * num_nodes))

    if num_to_prune > 0:
        prune_idx = np.argpartition(node_norms, num_to_prune)[:num_to_prune]
        
        # Zero outgoing weights (W2 rows)
        masks[index_to_prune][prune_idx, :] = 0.0

        # Zero incoming weights (W1 columns)
        input_idx = (index_to_prune - 1) % len(weight_matrices)
        masks[input_idx][:, prune_idx] = 0.0

    return masks


