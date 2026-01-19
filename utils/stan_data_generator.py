import numpy as np

def make_stan_data(model_name, task, X_train, y_train, X_test, args):
    """
    Returns a dictionary with Stan-compatible data fields,
    including prior-specific hyperparameters.
    """
    if model_name == "linreg_dirichlet_horseshoe" or model_name == "linreg_dirichlet_student_t" or model_name == "linreg_beta_student_t" or model_name == "linreg_beta_horseshoe":
        # y is a vector[N] in the Stan code
        stan_data = {
            "N": X_train.shape[0],
            "P": args.p,
            "X": X_train,
            "y": y_train,                     # 1D vector, no reshape
            "p_0": 2,
            "a": 2.0,
            "b": 4.0,
            "alpha": 0.1 * np.ones(args.p),
        }
        return stan_data
    
    if model_name == "linreg_gaussian":
        # y is a vector[N] in the Stan code
        stan_data = {
            "N": X_train.shape[0],
            "P": args.p,
            "X": X_train,
            "y": y_train,                     # 1D vector, no reshape
        }
        return stan_data
    
    if model_name in ["linreg_regularized_horseshoe", "linreg_regularized_horseshoe_scaled"]:
        stan_data = {
            "N": X_train.shape[0],
            "P": args.p,
            "X": X_train,
            "y": y_train,   # vector[N]
            "p_0": 4,
            "a": 2.0,
            "b": 4.0,
        }
        return stan_data


    # Shared basic data
    if task != "prior":
        stan_data = {
            'N': X_train.shape[0],
            'P': args.p,
            'X': X_train,
            'y': y_train.astype(int) if task == "classification" else y_train.reshape(-1, 1),
            'L': args.L,
            'H': args.H,
            'N_test': X_test.shape[0],
            'X_test': X_test,
            'output_nodes': args.num_classes
        }
    else:
        stan_data = {
        'N': X_train.shape[0],
        'P': args.p,
        'X': X_train,
        'L': args.L,
        'H': args.H,
        'N_test': X_test.shape[0],
        'X_test': X_test,
        'output_nodes': args.num_classes
        }

    # Model-specific extensions
    if model_name == "dirichlet_horseshoe" or model_name == "dirichlet_horseshoe_tanh" or model_name == "beta_horseshoe" or model_name == "beta_horseshoe_tanh":
        stan_data.update({
            'p_0': 4,
            'a': 2.0,
            'b': 4.0,
            'alpha': 0.1 * np.ones(args.p),
            'gamma': 1e-3,
        })
        print(stan_data['output_nodes'])
    
    elif model_name == "dirichlet_horseshoe_nodewise" or model_name == "dirichlet_horseshoe_tanh_nodewise" or model_name == "beta_horseshoe_nodewise" or model_name == "beta_horseshoe_tanh_nodewise":
        stan_data.update({
            'p_0': 4,
            'a': 2.0,
            'b': 4.0,
            'alpha': 0.1 * np.ones(args.p),
            'gamma': 1e-3,
        })

    elif model_name == "dirichlet_horseshoe_tanh_L2" or model_name == "dirichlet_student_t_tanh_L2" or model_name == "beta_horseshoe_tanh_L2" or model_name == "beta_student_t_tanh_L2":
        stan_data.update({
            'p_0': 4,
            'a': 2.0,
            'b': 4.0,
            'alpha': 0.1 * np.ones(args.p),
            'alpha_h': 0.1 * np.ones(args.H),
            'gamma': 1e-3,
        })
    
    elif model_name == "dirichlet_tau" or model_name == "dirichlet_tau_tanh" or model_name == "beta_tau" or model_name == "beta_tau_tanh":
        stan_data.update({
            'p_0': 4,
            'alpha': 0.1 * np.ones(args.p),
        })
    elif model_name == "dirichlet" or model_name == "dirichlet_tanh" or model_name == "beta" or model_name == "beta_tanh":
        stan_data.update({
            'alpha': 0.1 * np.ones(args.p),
        })
    
    elif model_name == "dirichlet_horseshoe_full":
        stan_data.update({
            'p_0': 4,
            'a': 2.0,
            'b': 4.0,
            'alpha': 0.1 * np.ones(args.p),
            'alpha_internal': 0.1 * np.ones(args.H),
            'gamma': 1e-3,
        })
        
    elif model_name == "dirichlet_student_t" or model_name == "dirichlet_student_t_tanh" or model_name == "beta_student_t" or model_name == "beta_student_t_tanh":
        stan_data.update({
            'alpha': 0.1 * np.ones(args.p),
            # Disse trengs når vi har med regulariseringen
            'p_0': 4,
            'a': 2.0,
            'b': 4.0,
        })
    
    elif model_name == "dirichlet_student_t_nodewise" or model_name == "dirichlet_student_t_tanh_nodewise" or model_name == "beta_student_t_nodewise" or model_name == "beta_student_t_tanh_nodewise":
        stan_data.update({
            'alpha': 0.1 * np.ones(args.p),
            # Disse trengs når vi har med regulariseringen
            'p_0': 4,
            'a': 2.0,
            'b': 4.0,
        })
        
    elif model_name == "dirichlet_gamma" or model_name == "dirichlet_gamma_tanh":
        stan_data.update({
            'alpha': 0.1 * np.ones(args.p),
            # Disse trengs når vi har med regulariseringen
            'p_0': 4,
            'a': 2.0,
            'b': 4.0,
        })

    elif model_name == "dirichlet_laplace":
        stan_data.update({
            'alpha': 0.1 * np.ones(args.p),
        })

    elif model_name == "gaussian" or model_name == "gaussian_tanh":
        # No additional priors needed
        pass
    
    elif model_name == "pcp" or model_name == "pcp_tanh":
        stan_data.update({
            'kappa_eps': 1e-10,
        })

    elif model_name == "horseshoe":
        stan_data.update({
            'tau': 0.5,
        })

    elif model_name == "regularized_horseshoe" or model_name == "regularized_horseshoe_tanh":
        stan_data.update({
            'p_0': 4,
            'a': 2.0,
            'b': 4.0,
        })

    elif model_name == "spike_and_slab":
        stan_data.update({
            'a': 0.1,
            'b': 0.1,
            'c': 1.0,
        })

    else:
        raise ValueError(f"No Stan data defined for model '{model_name}'")

    return stan_data

