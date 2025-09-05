import sys, os; sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__) if '__file__' in globals() else os.getcwd(), '..')))
import numpy as np
from generate_data import generate_Friedman_data, generate_two_moons

config = 1
output_dir = f"datasets/friedman/many"
os.makedirs(output_dir, exist_ok=True)

# Settings: (sample size, list of seeds)
# dataset_groups = {
#     100: [1, 2, 3, 4],#, 5, 6],
#     200: [7, 8, 9, 10],#, 11, 12],
#     #500: [13, 14, 15, 16, 17, 18],
#     #1000: [19]
# }

dataset_groups = {
    # 25: [1],
    # 50: [2],
    100: [3, 4, 5, 6],
    200: [7, 8, 9, 10],
    500: [11, 12, 13, 14, 15]
    # 500: [5],
    # 1000: [6]
}

D = 10        # Fixed number of features
#sigma_levels = [1.0, 3.0]#, 5.0]  # Noise levels
sigma_levels = 1 #0.2 #[1.0]
data_type = "Friedman"


for i, (N, seeds) in enumerate(dataset_groups.items()):
    for j, seed in enumerate(seeds):
        # Determine sigma based on index: 0–1 → 0.25, 2–3 → 0.5, 4–5 → 1.0
        sigma = sigma_levels#[j // 2]

        np.random.seed(seed)
        X_train, X_test, y_train, y_test = generate_Friedman_data(N=N, D=D, sigma=sigma)

        filename = f"{data_type}_N{N}_p{D}_sigma{sigma:.2f}_seed{seed}.npz"
        filepath = os.path.join(output_dir, filename)
        np.savez(filepath, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

        print(f"Saved: {filepath}")

