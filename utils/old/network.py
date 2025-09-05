"""
Utilities for extracting posterior mean weights from CmdStanPy fits,
based on a flexible layer structure specification.

Supports models with input-to-hidden, hidden-to-output, and optionally
hidden-to-hidden connections.
"""


import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

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

def extract_all_posterior_means(fits, layer_structure):
    """
    Extract posterior mean weights from all models in a dictionary of fits.

    Parameters:
        fits (dict): Dictionary of fits structured as
                    {model_name: {"posterior": CmdStanMCMC}}.
        layer_structure (dict): Layer structure as required by extract_posterior_means().

    Returns:
        dict: Dictionary of posterior means for each model,
            structured as {model_name: {param_name: mean_weights}}.
    """

    model_means = {}
    for name, fit_dict in fits.items():
        model_means[name] = extract_posterior_means(fit_dict[f"{name}"], layer_structure)
    return model_means

