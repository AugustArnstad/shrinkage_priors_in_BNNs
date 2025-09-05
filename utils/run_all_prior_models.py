import os
import numpy as np
import argparse
from model_runner import run_prior_model
from types import SimpleNamespace
import argparse
import re
from generate_data import generate_uniform_data


from cmdstanpy import set_cmdstan_path
set_cmdstan_path("/Users/augustarnstad/.cmdstan/cmdstan-2.36.0")

parser = argparse.ArgumentParser(description="Run regression model on multiple datasets.")

parser.add_argument("--model", type=str, required=True, help="Model name (e.g. dirichlet_horseshoe)")
parser.add_argument("--overwrite", action="store_true", help="If set, re-run even if result exists")
parser.add_argument("--warmup", type=int, default=1000, help="Max number of datasets to run")
parser.add_argument("--sample", type=int, default=1000, help="Max number of datasets to run")
parser.add_argument("--output_dir", type=str, default="results", help="Max number of datasets to run")
parser.add_argument("--standardize", action="store_true", help="Standardize data before fitting")

args = parser.parse_args()

#data_dir = f"datasets/friedman/many"
#X_train, X_test, y_train, y_test = generate_uniform_data()
path = "datasets/friedman/Friedman_N200_p10_sigma1.00_seed2.npz"
data = np.load(path)

X_train, X_test, y_train, y_test = data['X_train'], data['X_test'], data['y_train'], data['y_test']

warmup = args.warmup
sample = args.sample

N, p = X_train.shape
seed = 42
data_type = "prior"

config_name = f"dir_stud_t"
# Sett output-dir
model_output_dir = os.path.join(args.output_dir, args.model, config_name)


# Hopp over hvis resultatene allerede finnes
if not args.overwrite and os.path.exists(model_output_dir):
    print(f"[Skip] Already completed: {config_name}")
else:
    print(f"[Run] Running model on: {config_name}")

    run_prior_model(
        model_name=args.model,
        config_name=config_name,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        args=argparse.Namespace(
            N=N,
            p=p,
            sigma=None,
            data=data_type,
            standardize=args.standardize,
            test_shift=None,
            model=args.model,
            H=16,
            L=1,
            config=config_name,
            seed=seed,
            data_config="prior",
            model_output_dir=model_output_dir,
            burnin_samples=args.warmup,
            samples=args.sample,
        )
    )