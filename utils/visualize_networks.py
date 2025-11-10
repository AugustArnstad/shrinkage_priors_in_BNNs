import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from cmdstanpy import CmdStanMCMC


def extract_posterior_means(fit, layer_structure):
    """
    Extract posterior mean weights for all layers from a Stan model fit.

    Parameters:
        fit (CmdStanMCMC): Fitted Stan model containing sampled weights.
        layer_structure (dict): Dictionary specifying parameter names and shapes.
            Example structure:
                {
                    "input_to_hidden": {"name": "data_to_hidden_mat", "shape": (D, H)},
                    "hidden_to_output": {"name": "hidden_to_output", "shape": (H, 1)},
                    "hidden_to_hidden": {
                        "name": "hidden_hidden_weights",
                        "shape": [(H, H), (H, H)]
                    }  # optional
                }

    Returns:
        dict: Dictionary mapping each parameter name to its posterior mean weight matrix.
            Hidden-to-hidden layers are returned as a list of 2D arrays (one per layer).
    """

    means = {}

    # Input-to-hidden
    param = layer_structure['input_to_hidden']['name']
    shape = layer_structure['input_to_hidden']['shape']
    means[param] = fit.stan_variable(param).mean(axis=0)#.reshape(shape)

    # Optional hidden-to-hidden
    if 'hidden_to_hidden' in layer_structure:
        param = layer_structure['hidden_to_hidden']['name']
        shapes = layer_structure['hidden_to_hidden']['shape']
        raw = fit.stan_variable(param).mean(axis=0)
        means[param] = [raw[i].reshape(shapes[i]) for i in range(len(shapes))]

    # Hidden-to-output
    param = layer_structure['hidden_to_output']['name']
    shape = layer_structure['hidden_to_output']['shape']
    means[param] = fit.stan_variable(param).mean(axis=0).reshape(shape)

    return means

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

def plot_all_networks_subplots_activations(model_dicts, layer_sizes, node_activation_colors=None, activation_color_max=None, max_width=5.0, ncols=3, figsize_per_plot=(5, 4), signed_colors=False):
    """
    Plot multiple neural networks as subplots, with edge thickness representing weight magnitude
    and hidden node color intensity representing activation frequency.

    Parameters:
        model_dicts (dict): Dictionary mapping model names to weight dicts.
                            Each weight dict must include:
                            - 'W_1': input-to-hidden weights (P, H)
                            - 'W_L': hidden-to-output weights (H, 1)
                            - optionally 'W_internal': list of (H, H) hidden-hidden weights
        layer_sizes (list[int]): List of node counts for each layer (e.g. [8, 16, 1]).
        node_activation_colors (dict): Maps model name to array of activation frequencies for hidden nodes.
        max_width (float): Maximum line width for strongest edge. Default is 5.0.
        ncols (int): Number of subplot columns. Default is 3.
        figsize_per_plot (tuple): Base figure size per subplot (width, height).
        signed_colors (bool): If True, positive weights are red and negative weights are blue.

    Returns:
        fig, edge_widths: Matplotlib figure and list of edge widths for the last model.
    """

    n_models = len(model_dicts)
    nrows = int(np.ceil(n_models / ncols))
    figsize = (figsize_per_plot[0] * ncols, figsize_per_plot[1] * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()

    for ax in axes[n_models:]:
        ax.axis('off')

    for idx, (title, weight_dict) in enumerate(model_dicts.items()):
        G = nx.DiGraph()
        pos = {}
        node_ids_per_layer = []
        node_colors = []

        # Add nodes
        for layer_idx, size in enumerate(layer_sizes):
            nodes = []
            y_coords = np.linspace(size - 1, 0, size) - (size - 1) / 2
            for i in range(size):
                nid = f"L{layer_idx}_{i}"
                G.add_node(nid)
                pos[nid] = (layer_idx, y_coords[i])
                nodes.append(nid)

                # Set node color
                # if node_activation_colors and layer_idx == 1:
                #     act_freqs = node_activation_colors.get(title, np.zeros(size))
                #     color_val = np.clip(act_freqs[i], 0.0, 1.0)  # Ensure valid range
                #     color = plt.cm.winter(color_val)
                if node_activation_colors and layer_idx == 1:
                    act_freqs = node_activation_colors.get(title, np.zeros(size))
                    scale = activation_color_max if activation_color_max is not None else 1.0
                    color_val = np.clip(act_freqs[i] / scale, 0.0, 1.0)  # Normalize globally
                    color = plt.cm.winter(color_val)
                else:
                    color = 'lightgray'
                node_colors.append(color)

            node_ids_per_layer.append(nodes)

        edge_colors = []
        edge_widths = []

        # Function to add edges from W
        def add_edges(W, in_nodes, out_nodes, tol=1e-10):
            for j, out_node in enumerate(out_nodes):
                for i, in_node in enumerate(in_nodes):
                    w = W[i, j]
                    if abs(w)<tol:
                        continue
                    G.add_edge(in_node, out_node, weight=abs(w))
                    edge_colors.append('red' if w >= 0 else 'blue')
                    edge_widths.append(abs(w))

        # Input-to-hidden
        add_edges(weight_dict['W_1'], node_ids_per_layer[0], node_ids_per_layer[1])

        # Hidden-to-hidden
        if 'W_internal' in weight_dict:
            for l in range(len(weight_dict['W_internal'])):
                add_edges(weight_dict['W_internal'][l], node_ids_per_layer[l+1], node_ids_per_layer[l+2])

        # Hidden-to-output
        add_edges(weight_dict['W_L'], node_ids_per_layer[-2], node_ids_per_layer[-1])

        #labels = {nid: nid for nid in G.nodes}
        #nx.draw_networkx_labels(G, pos, labels=labels, ax=axes[idx], font_size=8)

        #edge_widths = [G[u][v]['weight'] for u, v in G.edges()]
        edge_widths = np.array([G[u][v]['weight'] for u, v in G.edges()])
        edge_widths = np.clip(edge_widths / (edge_widths.max() + 1e-12) * max_width, 0.5, None)

        nx.draw(G, pos, ax=axes[idx], node_color=node_colors,
                edge_color=edge_colors if signed_colors else 'red',
                width=edge_widths, with_labels=False,
                node_size=400, arrows=False)

        axes[idx].set_title(title, fontsize=10)
        axes[idx].axis('off')

    plt.tight_layout()
    return fig, edge_widths

def get_pruned_mean_weights(fit, layer_structure, sparsity_level):
    """
    Extract posterior mean weights from a single model and prune the input layer.

    Parameters:
        fit : CmdStanMCMC
        layer_structure : dict
            Structure of layers with param names and shapes.
        sparsity_level : float
            Proportion of weights to prune from input-to-hidden layer.

    Returns:
        dict: pruned mean weights for plotting.
    """
    means = extract_posterior_means(fit, layer_structure)
    # Prune input-to-hidden layer only (you can generalize if needed)
    W1 = means['W_1']
    P, H = W1.shape
    num_to_prune = int(np.floor(sparsity_level * P * H))
    mask = prune_weights(W1, num_to_prune)
    W1_pruned = W1 * mask

    means['W_1'] = W1_pruned
    return means

def extract_all_pruned_means(fits, layer_structure, sparsity_level):
    """
    Apply pruning to all fits and return dict of pruned weight dictionaries.

    Parameters:
        fits : dict of {model_name: {"posterior": CmdStanMCMC}}
        layer_structure : as used in plotting
        sparsity_level : float

    Returns:
        dict: {model_name: {pruned weights}}
    """
    model_means = {}
    for name, fit_dict in fits.items():
        pruned_means = get_pruned_mean_weights(fit_dict["posterior"], layer_structure, sparsity_level)
        model_means[name] = pruned_means
    return model_means

def compute_activation_frequency(fit, x_train):
    W1 = fit['posterior'].stan_variable('W_1').mean(axis=0)
    b1 = fit['posterior'].stan_variable('hidden_bias').mean(axis=0)
    pre_act = x_train @ W1 + b1
    post_act = np.maximum(0, pre_act)
    return (post_act > 0).mean(axis=0)  # shape: (H,)

def prune_nodes_posterior_means(fit, layer_structure, num_nodes_to_prune):
    """
    Prune entire nodes (neurons) from posterior mean weights.

    Parameters:
        fit : CmdStanMCMC
        layer_structure : dict with parameter names and shapes
        num_nodes_to_prune : int, number of hidden nodes to prune

    Returns:
        dict: pruned mean weights for plotting
    """
    means = extract_posterior_means(fit, layer_structure)

    W1 = means['W_1']  # shape (P, H)
    W2 = means['W_L']  # shape (H, O)
    
    # Compute node importance: use L2 norm of outgoing weights
    node_norms = np.linalg.norm(W2, axis=1)  # shape (H,)

    # Identify least important nodes
    prune_idx = np.argpartition(node_norms, num_nodes_to_prune)[:num_nodes_to_prune]

    # Prune incoming weights (columns of W1) and outgoing weights (rows of W2)
    W1_mask = np.ones_like(W1)
    W1_mask[:, prune_idx] = 0.0
    W2_mask = np.ones_like(W2)
    W2_mask[prune_idx, :] = 0.0

    means['W_1'] = W1 * W1_mask
    means['W_L'] = W2 * W2_mask

    return means

def extract_all_pruned_node_means(fits, layer_structure, num_nodes_to_prune):
    """
    Apply node pruning to all fits.

    Parameters:
        fits : dict of {model_name: {"posterior": CmdStanMCMC}}
        layer_structure : as above
        num_nodes_to_prune : int

    Returns:
        dict: {model_name: {pruned weights}}
    """
    model_means = {}
    for name, fit_dict in fits.items():
        pruned_means = prune_nodes_posterior_means(fit_dict["posterior"], layer_structure, num_nodes_to_prune)
        model_means[name] = pruned_means
    return model_means
