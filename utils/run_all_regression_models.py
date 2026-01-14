import os
import numpy as np
import argparse
from model_runner import run_regression_model
from types import SimpleNamespace
import argparse
import re


from cmdstanpy import set_cmdstan_path
set_cmdstan_path("/Users/augustarnstad/.cmdstan/cmdstan-2.36.0")

parser = argparse.ArgumentParser(description="Run regression model on multiple datasets.")

parser.add_argument("--model", type=str, required=True, help="Model name (e.g. dirichlet_horseshoe)")
parser.add_argument("--pattern", type=str, default=".*", help="Regex pattern to match dataset filenames")
parser.add_argument("--overwrite", action="store_true", help="If set, re-run even if result exists")
parser.add_argument("--N", type=int, nargs='*', help="Only include datasets with these sample sizes (e.g. --N 100 500)")
parser.add_argument("--sigma", type=float, nargs='*', help="Only include datasets with these noise levels")
parser.add_argument("--GAM_config", type=int, default=1, help="Type of GAM structure to use (1, 2 or 3)")
parser.add_argument("--warmup", type=int, default=1000, help="Max number of datasets to run")
parser.add_argument("--sample", type=int, default=1000, help="Max number of datasets to run")
parser.add_argument("--limit", type=int, default=None, help="Max number of datasets to run")
parser.add_argument("--output_dir", type=str, default="results", help="Max number of datasets to run")

args = parser.parse_args()

#data_dir = f"datasets/friedman"
data_dir = f"datasets/friedman"
results_dir = args.output_dir 

warmup = args.warmup
sample = args.sample
H = 16 
L = 1
files = sorted(f for f in os.listdir(data_dir) if f.endswith(".npz") and re.match(args.pattern, f))
processed = 0

for fname in files:
    tokens = fname.replace(".npz", "").split("_")
    data_type = tokens[0]
    N = int(tokens[1][1:])
    p = int(tokens[2][1:])
    sigma = float(tokens[3][5:])
    seed = int(tokens[4][4:])

    # Filter by --N
    if args.N and N not in args.N:
        continue
    # Filter by --sigma
    if args.sigma and sigma not in args.sigma:
        continue
    # Apply --limit
    if args.limit is not None and processed >= args.limit:
        break

    path = os.path.join(data_dir, fname)
    data = np.load(path)

    config_name = fname.replace(".npz", "")
    model_output_dir = os.path.join(results_dir, args.model, config_name)

    if not args.overwrite and os.path.exists(model_output_dir):
        print(f"[Skip] Already completed: {config_name}")
        continue

    print(f"[Run] Running model on: {fname}")

    run_regression_model(
        model_name=args.model,
        config_name=config_name,
        X_train=data['X_train'],
        X_test=data['X_test'],
        y_train=data['y_train'],
        y_test=data['y_test'],
        args=argparse.Namespace(
            N=N,
            p=p,
            sigma=sigma,
            data=data_type,
            standardize=True,
            test_shift=None,
            model=args.model,
            H=H,
            L=L,
            config=config_name,
            seed=seed,
            data_config = "nodewise_lambda",
            model_output_dir = model_output_dir,
            burnin_samples=warmup,
            samples=sample,
        )
    )
    processed += 1

