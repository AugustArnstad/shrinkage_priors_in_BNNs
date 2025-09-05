import os
import numpy as np
import argparse
from model_runner import run_classification_model
from types import SimpleNamespace
from cmdstanpy import set_cmdstan_path
from generate_data import load_wine_quality_data

# Sett CmdStan-sti
set_cmdstan_path("/Users/augustarnstad/.cmdstan/cmdstan-2.36.0")

# Argumentparser
parser = argparse.ArgumentParser(description="Run classification model on Breast Cancer dataset.")
parser.add_argument("--model", type=str, required=True, help="Model name (e.g. dirichlet_horseshoe)")
parser.add_argument("--overwrite", action="store_true", help="If set, re-run even if result exists")
parser.add_argument("--output_dir", type=str, default="results", help="Output directory for results")
parser.add_argument("--standardize", action="store_true", help="Standardize data before fitting")

args = parser.parse_args()

# Last inn data
X_train, X_test, y_train, y_test, *_ = load_wine_quality_data(
    test_size=0.2, standardize=args.standardize, random_state=42
)

N, p = X_train.shape
seed = 42
data_type = "wine"
classes = len(np.unique(y_train))
if args.standardize:
    config_name = f"wine_N{N}_p{p}_standardized"
else:     
    config_name = f"wine_N{N}_p{p}"

model_output_dir = os.path.join(args.output_dir, args.model, config_name)

# Hopp over hvis resultater finnes
if not args.overwrite and os.path.exists(model_output_dir):
    print(f"[Skip] Already completed: {config_name}")
else:
    print(f"[Run] Running model on: {config_name}")

    run_classification_model(
        model_name=args.model,
        config_name=config_name,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        args=argparse.Namespace(
            N=N,
            p=p,
            num_classes=classes,
            data=data_type,
            standardize=args.standardize,
            test_shift=None,
            model=args.model,
            H=16,
            L=1,
            config=config_name,
            seed=seed,
            data_config="realworld",
            model_output_dir=model_output_dir,
            burnin_samples=1000,
            samples=1000,
        )
    )
