# utils/model_runner.py
def run_regression_model(model_name, config_name, X_train, X_test, y_train, y_test, args):
    from cmdstanpy import CmdStanModel
    from stan_data_generator import make_stan_data
    from io_helpers import save_metadata
    import os, shutil
    import numpy as np
    
    # Set seed for reproducibility if provided
    seed = getattr(args, 'seed', None)
    if seed is not None:
        np.random.seed(seed)

    task = "regression"
    args.num_classes = 1  # Still needed

    stan_data = make_stan_data(model_name, task, X_train, y_train, X_test, args)

    model_path = f"bnn_regression_models/{model_name}.stan"
    model = CmdStanModel(stan_file=model_path, force_compile=True)

    fit = model.sample(
        data=stan_data,
        chains=4,
        iter_sampling=args.samples,
        iter_warmup=args.burnin_samples,
        adapt_delta=0.90,
        parallel_chains=4,
        show_console=False,
        max_treedepth = 12,
    )
    
    if args.data_config == "uci": 
        if args.standardize:
            output_dir = os.path.join(
            args.model_output_dir, "standardized"
        )
        else:
            output_dir = os.path.join(
                args.model_output_dir
            )
    else:
        output_dir = os.path.join(
            args.model_output_dir
        )
    os.makedirs(output_dir, exist_ok=True)
    save_metadata(output_dir, args, config_name)

    for i, path in enumerate(fit.runset.csv_files, start=1):
        shutil.copy(path, os.path.join(output_dir, f"chain_{i}.csv"))

    print(f"[✓] Saved results to: {output_dir}")


def run_classification_model(model_name, config_name, X_train, X_test, y_train, y_test, args):
    from cmdstanpy import CmdStanModel
    from stan_data_generator import make_stan_data
    from io_helpers import save_metadata
    import os, shutil
    import numpy as np
    
    # Set seed for reproducibility if provided
    seed = getattr(args, 'seed', None)
    if seed is not None:
        np.random.seed(seed)

    task = "classification"

    stan_data = make_stan_data(model_name, task, X_train, y_train, X_test, args)

    model_path = f"bnn_classification_models/{model_name}.stan"
    model = CmdStanModel(stan_file=model_path, force_compile=True)

    fit = model.sample(
        data=stan_data,
        chains=4,
        iter_sampling=args.samples,
        iter_warmup=args.burnin_samples,
        adapt_delta=0.8,
        parallel_chains=4,
        show_console=False,
        #max_treedepth = 12,
    )
    
    if args.data_config == "uci": 
        if args.standardize:
            output_dir = os.path.join(
            args.model_output_dir, "standardized"
        )
        else:
            output_dir = os.path.join(
                args.model_output_dir
            )
    else:
        output_dir = os.path.join(
            args.model_output_dir
        )
    os.makedirs(output_dir, exist_ok=True)
    save_metadata(output_dir, args, config_name)

    for i, path in enumerate(fit.runset.csv_files, start=1):
        shutil.copy(path, os.path.join(output_dir, f"chain_{i}.csv"))

    print(f"[✓] Saved results to: {output_dir}")


def run_prior_model(model_name, config_name, X_train, X_test, y_train, args):
    from cmdstanpy import CmdStanModel
    from stan_data_generator import make_stan_data
    from io_helpers import save_metadata
    import os, shutil
    import numpy as np
    
    # Set seed for reproducibility if provided
    seed = getattr(args, 'seed', None)
    if seed is not None:
        np.random.seed(seed)

    task = "prior"
    args.num_classes = 1  # Still needed

    stan_data = make_stan_data(model_name, task, X_train, y_train, X_test, args)

    model_path = f"bnn_prior_models/{model_name}.stan"
    model = CmdStanModel(stan_file=model_path, force_compile=True)

    fit = model.sample(
        data=stan_data,
        chains=4,
        iter_sampling=args.samples,
        iter_warmup=args.burnin_samples,
        adapt_delta=0.8,
        parallel_chains=4,
        show_console=False,
        #max_treedepth = 12,
    )
    
    if args.data_config == "uci": 
        if args.standardize:
            output_dir = os.path.join(
            args.model_output_dir, "standardized"
        )
        else:
            output_dir = os.path.join(
                args.model_output_dir
            )
    else:
        output_dir = os.path.join(
            args.model_output_dir
        )
    os.makedirs(output_dir, exist_ok=True)
    save_metadata(output_dir, args, config_name)

    for i, path in enumerate(fit.runset.csv_files, start=1):
        shutil.copy(path, os.path.join(output_dir, f"chain_{i}.csv"))

    print(f"[✓] Saved results to: {output_dir}")
