import os
import glob
from cmdstanpy import from_csv

def load_fit(model_glob_path):
    files = sorted(glob.glob(model_glob_path))
    if not files:
        print(f"[WARNING] No files matched: {model_glob_path}")
        return None
    return from_csv(path=files, method='sample')

def get_model_fits(config, results_dir, models=None, include_prior=True, include_posterior=True):
    """
    Load CmdStanPy fits for selected models and config.

    Parameters:
        config (str): Subfolder under each model's csv_files path.
        models (list[str], optional): Subset of models to load. Defaults to all.
        include_prior (bool): Whether to load prior fits.
        include_posterior (bool): Whether to load posterior fits.

    Returns:
        dict: {model_name: {"prior": fit or None, "posterior": fit or None}}
    """

    all_model_paths = {
        "Spike & Slab": ('spike_and_slab_prior', 'spike_and_slab'),
        "Gaussian": ('gaussian_prior', 'gaussian'),
        "Horseshoe": ('horseshoe_prior', 'horseshoe'),
        "Regularized Horseshoe": ('regularized_horseshoe_prior', 'regularized_horseshoe'),
        "Dirichlet Laplace": ('dirichlet_laplace_prior', 'dirichlet_laplace'),
        "Dirichlet Horseshoe": ('dirichlet_horseshoe_prior', 'dirichlet_horseshoe'),
        "Dirichlet Horseshoe full": ('dirichlet_horseshoe_prior_full', 'dirichlet_horseshoe_full'),
        "Dirichlet Student T": ('dirichlet_student_t_prior', 'dirichlet_student_t'),
        "Beta Horseshoe": ('beta_horseshoe_prior', 'beta_horseshoe'),
        "Beta Student T": ('beta_student_t_prior', 'beta_student_t'),
        "Dirichlet tau": ('dirichlet_tau_prior', 'dirichlet_tau'),
        "Beta tau": ('beta_tau_prior', 'beta_tau'),
        "Dirichlet": ('dirichlet_prior', 'dirichlet'),
        "Beta": ('beta_prior', 'beta'),
        "Dirichlet Gamma": ('dirichlet_gamma_prior', 'dirichlet_gamma'),
        "Pred CP": ('pcp_prior', 'pcp'),
        
        "Gaussian tanh": ('gaussian_tanh_prior', 'gaussian_tanh'),
        "Regularized Horseshoe tanh": ('regularized_horseshoe_tanh_prior', 'regularized_horseshoe_tanh'),
        "Dirichlet Horseshoe tanh": ('dirichlet_horseshoe_tanh_prior', 'dirichlet_horseshoe_tanh'),
        "Dirichlet Student T tanh": ('dirichlet_student_t_tanh_prior', 'dirichlet_student_t_tanh'),
        "Dirichlet Horseshoe tanh L2": ('dirichlet_horseshoe_tanh_L2_prior', 'dirichlet_horseshoe_tanh_L2'),
        "Dirichlet Student T tanh L2": ('dirichlet_student_t_tanh_L2_prior', 'dirichlet_student_t_tanh_L2'),
        "Beta Horseshoe tanh L2": ('beta_horseshoe_tanh_L2_prior', 'beta_horseshoe_tanh_L2'),
        "Beta Student T tanh L2": ('beta_student_t_tanh_L2_prior', 'beta_student_t_tanh_L2'),
        "Dirichlet Gamma tanh": ('dirichlet_gamma_prior_tanh', 'dirichlet_gamma_tanh'),
        "Beta Horseshoe tanh": ('beta_horseshoe_tanh_prior', 'beta_horseshoe_tanh'),
        "Beta Student T tanh": ('beta_student_t_tanh_prior', 'beta_student_t_tanh'),
        "Dirichlet tau tanh": ('dirichlet_tau_tanh_prior', 'dirichlet_tau_tanh'),
        "Beta tau tanh": ('beta_tau_tanh_prior', 'beta_tau_tanh'),
        "Dirichlet tanh": ('dirichlet_tanh_prior', 'dirichlet_tanh'),
        "Beta tanh": ('beta_tanh_prior', 'beta_tanh'),
        "Pred CP tanh": ('pcp_prior_tanh','pcp_tanh'),
        
        "Regularized Horseshoe tanh nodewise": ('regularized_horseshoe_tanh_nodewise_prior', 'regularized_horseshoe_tanh_nodewise'),
        "Dirichlet Horseshoe tanh nodewise": ('dirichlet_horseshoe_tanh_nodewise_prior', 'dirichlet_horseshoe_tanh_nodewise'),
        "Dirichlet Student T tanh nodewise": ('dirichlet_student_t_tanh_nodewise_prior', 'dirichlet_student_t_tanh_nodewise'),
        "Beta Horseshoe tanh nodewise": ('beta_horseshoe_tanh_nodewise_prior', 'beta_horseshoe_tanh_nodewise'),
        "Beta Student T tanh nodewise": ('beta_student_t_tanh_nodewise_prior', 'beta_student_t_tanh_nodewise'),
        
        "Linreg Gaussian": ('linreg_gaussian_prior', 'linreg_gaussian'),                
        "Linreg Regularized Horseshoe": ('linreg_regularized_horseshoe_prior', 'linreg_regularized_horseshoe'),
        "Linreg Regularized Horseshoe scaled": ('linreg_regularized_horseshoe_scaled_prior', 'linreg_regularized_horseshoe_scaled'),
        "Linreg Dirichlet Horseshoe": ('linreg_dirichlet_horseshoe_prior', 'linreg_dirichlet_horseshoe'),
        "Linreg Dirichlet Student T": ('linreg_dirichlet_student_t_prior', 'linreg_dirichlet_student_t'),
                
        "Linreg Beta Horseshoe": ('linreg_beta_horseshoe_prior', 'linreg_beta_horseshoe'),
        "Linreg Beta Student T": ('linreg_beta_student_t_prior', 'linreg_beta_student_t'),
        
        "Linreg Regularized Horseshoe Centered": ('linreg_regularized_horseshoe_prior', 'linreg_regularized_horseshoe_centered'),
        "Linreg Dirichlet Horseshoe Centered": ('linreg_dirichlet_horseshoe_prior', 'linreg_dirichlet_horseshoe_centered'),
        "Linreg Dirichlet Student T Centered": ('linreg_dirichlet_student_t_prior', 'linreg_dirichlet_student_t_centered'),
    }

    # Filter models if a subset is provided
    if models is not None:
        all_model_paths = {k: v for k, v in all_model_paths.items() if k in models}

    fits = {}

    for model_name, (prior_dir, post_dir) in all_model_paths.items():
        entry = {}

        if include_prior:
            prior_path = os.path.join(results_dir, prior_dir, config, "chain_*.csv")
            prior_fit = load_fit(prior_path)
            if prior_fit is not None:
                entry["prior"] = prior_fit

        if include_posterior:
            post_path = os.path.join(results_dir, post_dir, config, "chain_*.csv")
            post_fit = load_fit(post_path)
            if post_fit is not None:
                entry["posterior"] = post_fit

        if entry:  # Only add if at least one fit was found
            fits[model_name] = entry

    return fits



def extract_weights(fit, var_name="data_to_hidden"):
    return fit.stan_variable(var_name)


