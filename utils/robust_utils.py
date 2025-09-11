import numpy as np
import pandas as pd
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class StanNNRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim, activation=torch.tanh):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, 1)
        self.activation = activation

    def forward(self, x):
        x = self.activation(self.linear1(x))
        return self.linear2(x).squeeze(-1)  # logits for regression


class StanNNClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, activation=torch.tanh):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        self.activation = activation

    def forward(self, x):
        x = self.activation(self.linear1(x))
        return self.linear2(x)  # logits, softmax tas utenfor ved behov


def build_pytorch_model_from_stan_sample(
    fit,
    sample_idx,
    input_dim,
    hidden_dim,
    output_dim=None,
    task="classification",
    activation=torch.tanh
):
    """
    Construct a PyTorch model using weights from a single Stan sample.
    Matches Stan shapes:
      W_1: (input_dim, hidden_dim)  == (D, H)
      W_L: (hidden_dim, output_dim) == (H, O)  [klassifisering]
           (hidden_dim,)            == (H,)    [regresjon]
      hidden_bias: (H,)
      output_bias: (O,) eller skalar (regresjon)
    """
    # Hent parametre fra Stan
    W1 = fit.stan_variable("W_1")[sample_idx]                  # (D, H) eller (H, D) avh.
    b1 = fit.stan_variable("hidden_bias")[sample_idx]          # (H,) eller (1, H)

    # Sørg for 1D bias
    import numpy as np
    b1 = np.asarray(b1).reshape(-1)

    # Bygg modell
    if task == "regression":
        W2 = fit.stan_variable("W_L")[sample_idx]              # (H,)
        b2 = fit.stan_variable("output_bias")[sample_idx]      # skalar
        model = StanNNRegressor(input_dim, hidden_dim, activation=activation)

        with torch.no_grad():
            # W1 til (H, D)
            if W1.shape == (input_dim, hidden_dim):
                W1_t = torch.tensor(W1, dtype=torch.float32).T
            elif W1.shape == (hidden_dim, input_dim):
                W1_t = torch.tensor(W1, dtype=torch.float32)
            else:
                raise ValueError(f"Uventet W_1-shape: {W1.shape}")

            model.linear1.weight.copy_(W1_t)                                   # (H, D)
            model.linear1.bias.copy_(torch.tensor(b1, dtype=torch.float32))     # (H,)
            model.linear2.weight.copy_(torch.tensor(W2, dtype=torch.float32).unsqueeze(0))  # (1, H)
            model.linear2.bias.copy_(torch.tensor([b2], dtype=torch.float32))   # (1,)
        return model

    elif task == "classification":
        W2 = fit.stan_variable("W_L")[sample_idx]              # (H, O)
        b2 = fit.stan_variable("output_bias")[sample_idx]      # (O,)
        model = StanNNClassifier(input_dim, hidden_dim, output_dim, activation=activation)

        with torch.no_grad():
            # W1 til (H, D)
            if W1.shape == (input_dim, hidden_dim):
                W1_t = torch.tensor(W1, dtype=torch.float32).T
            elif W1.shape == (hidden_dim, input_dim):
                W1_t = torch.tensor(W1, dtype=torch.float32)
            else:
                raise ValueError(f"Uventet W_1-shape: {W1.shape}")

            model.linear1.weight.copy_(W1_t)                                   # (H, D)
            model.linear1.bias.copy_(torch.tensor(b1, dtype=torch.float32))     # (H,)
            model.linear2.weight.copy_(torch.tensor(W2, dtype=torch.float32).T) # (O, H)
            model.linear2.bias.copy_(torch.tensor(b2, dtype=torch.float32))     # (O,)
        return model

    else:
        raise ValueError("task must be either 'regression' or 'classification'")


def fgsm_attack(model, x, y_true, epsilon):
    """
    Generates an adversarial example using FGSM.
    model: PyTorch model (returns logits)
    x: input tensor (requires_grad=True)  shape (1, D)
    y_true: target class index tensor     shape (1,)
    epsilon: perturbation magnitude
    """
    x_adv = x.clone().detach().requires_grad_(True)
    logits = model(x_adv)                    # logits
    loss = F.cross_entropy(logits, y_true)   # CE forventer logits
    loss.backward()
    perturbation = epsilon * x_adv.grad.sign()
    x_adv = x_adv + perturbation
    return x_adv.detach()


def estimate_probabilistic_robustness(
    fit,
    x_test_sample,
    y_true,
    sample_indices,
    input_dim,
    hidden_dim,
    output_dim,
    epsilon=0.1,
    delta=0.1,
    p_norm=1,
    activation=torch.tanh
):
    """
    Estimate the probabilities p1 (softmax shift > delta) and p2 (label change)
    under the posterior, using FGSM in input space.
    """
    import numpy as np
    import tqdm

    p_1_flags = []
    p_2_flags = []

    x_test_tensor = torch.tensor(x_test_sample, dtype=torch.float32)

    for idx in tqdm.tqdm(sample_indices, disable=True):
        model = build_pytorch_model_from_stan_sample(
            fit=fit,
            sample_idx=idx,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            task="classification",
            activation=activation
        )

        model.eval()
        x_tensor = x_test_tensor.clone().detach().unsqueeze(0).requires_grad_(True)  # (1, D)
        y_tensor = torch.tensor([y_true], dtype=torch.long)                           # (1,)

        # FGSM
        x_adv = fgsm_attack(model, x_tensor, y_tensor, epsilon)

        # Sannsynligheter før/etter (softmax på logits)
        probs_orig = F.softmax(model(x_tensor), dim=1)
        probs_adv  = F.softmax(model(x_adv),  dim=1)

        # p1: softmax-shift
        shift = torch.norm(probs_orig - probs_adv, p=p_norm).item()
        is_softmax_shift = (shift > delta)

        # p2: label change
        label_orig = torch.argmax(probs_orig, dim=1).item()
        label_adv  = torch.argmax(probs_adv,  dim=1).item()
        is_label_changed = (label_orig != label_adv)

        p_1_flags.append(is_softmax_shift)
        p_2_flags.append(is_label_changed)

    p_1_prob = float(np.mean(p_1_flags))
    p_2_prob = float(np.mean(p_2_flags))
    return p_1_prob, p_1_flags, p_2_prob, p_2_flags


def estimate_robustness_for_all_models(
    fits_dict,
    x_test_sample,
    y_true,
    input_dim,
    hidden_dim,
    output_dim,
    sample_indices,
    epsilon=0.1,
    delta=0.1,
    p_norm=1,
    activation=torch.tanh
):
    """
    Compute probabilistic robustness estimates (p₁ and p₂) for each model in a fit dictionary.
    """
    results = {}
    for model_name, model_entry in fits_dict.items():
        fit = model_entry["posterior"]
        p1_prob, _, p2_prob, _ = estimate_probabilistic_robustness(
            fit=fit,
            x_test_sample=x_test_sample,
            y_true=y_true,
            sample_indices=sample_indices,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            epsilon=epsilon,
            delta=delta,
            p_norm=p_norm,
            activation=activation
        )
        results[model_name] = {"p1": p1_prob, "p2": p2_prob}
    return results


def estimate_robustness_over_test_set(
    x_test,
    y_test,
    fits_dict,
    input_dim,
    hidden_dim,
    output_dim,
    sample_indices,
    epsilon=0.1,
    delta=0.1,
    p_norm=1,
    activation=torch.tanh,
):
    import pandas as pd
    import tqdm

    all_results = []
    for i in tqdm.tqdm(range(len(x_test)), desc="Test samples", disable=False):
        x_i = x_test.iloc[i].to_numpy()
        y_i = int(y_test.iloc[i])  # forventes 0-indeksert

        robustness_results = estimate_robustness_for_all_models(
            fits_dict=fits_dict,
            x_test_sample=x_i,
            y_true=y_i,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            sample_indices=sample_indices,
            epsilon=epsilon,
            delta=delta,
            p_norm=p_norm,
            activation=activation
        )

        for model, robustness in robustness_results.items():
            all_results.append({
                "epsilon": epsilon,
                "model": model,
                "robustness": robustness
            })

    df_results = pd.DataFrame(all_results)
    return df_results


# class StanNNRegressor(nn.Module):
#     def __init__(self, input_dim, hidden_dim):
#         super().__init__()
#         self.linear1 = nn.Linear(input_dim, hidden_dim)
#         self.linear2 = nn.Linear(hidden_dim, 1)

#     def forward(self, x):
#         x = F.relu(self.linear1(x))
#         return self.linear2(x).squeeze(-1)


# class StanNNClassifier(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim):
#         super().__init__()
#         self.linear1 = nn.Linear(input_dim, hidden_dim)
#         self.linear2 = nn.Linear(hidden_dim, output_dim)

#     def forward(self, x):
#         x = F.relu(self.linear1(x))
#         return self.linear2(x)
    


# class StanNNRegressor(nn.Module):
#     def __init__(self, input_dim, hidden_dim, activation=F.relu):
#         super().__init__()
#         self.linear1 = nn.Linear(input_dim, hidden_dim)
#         self.linear2 = nn.Linear(hidden_dim, 1)
#         self.activation = activation

#     def forward(self, x):
#         x = self.activation(self.linear1(x))
#         return self.linear2(x).squeeze(-1)


# class StanNNClassifier(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim, activation=F.relu):
#         super().__init__()
#         self.linear1 = nn.Linear(input_dim, hidden_dim)
#         self.linear2 = nn.Linear(hidden_dim, output_dim)
#         self.activation = activation

#     def forward(self, x):
#         x = self.activation(self.linear1(x))
#         return self.linear2(x)



# def build_pytorch_model_from_stan_sample(
#     fit,
#     sample_idx,
#     input_dim,
#     hidden_dim,
#     output_dim=None,
#     task="classification",
#     activation = F.relu
# ):
#     """
#     Construct a PyTorch model using weights from a single Stan sample.
#     """
#     W1 = fit.stan_variable("W_1")[sample_idx]  # (H, D)
#     b1 = fit.stan_variable("hidden_bias")[sample_idx].squeeze(0)   # (H,)
#     if task == "regression":
#         W2 = fit.stan_variable("W_L")[sample_idx]  # (H,)
#         b2 = fit.stan_variable("output_bias")[sample_idx]       # scalar
#         model = StanNNRegressor(input_dim, hidden_dim, activation=activation)
#         with torch.no_grad():
#             model.linear1.weight.copy_(torch.tensor(W1))
#             model.linear1.bias.copy_(torch.tensor(b1))
#             model.linear2.weight.copy_(torch.tensor(W2).unsqueeze(0))  # (1, H)
#             model.linear2.bias.copy_(torch.tensor([b2]))
#     elif task == "classification":
#         W2 = fit.stan_variable("W_L")[sample_idx]  # (H, output_dim)
#         b2 = fit.stan_variable("output_bias")[sample_idx]       # (output_dim,)
#         model = StanNNClassifier(input_dim, hidden_dim, output_dim, activation=activation)
#         with torch.no_grad():
#             model.linear1.weight.copy_(torch.tensor(W1))
#             model.linear1.bias.copy_(torch.tensor(b1))
#             model.linear2.weight.copy_(torch.tensor(W2).T)  # PyTorch expects (out, in)
#             model.linear2.bias.copy_(torch.tensor(b2))
#     else:
#         raise ValueError("task must be either 'regression' or 'classification'")

#     return model


# def fgsm_attack(model, x, y_true, epsilon):
#     """
#     Generates an adversarial example using FGSM.
#     model: PyTorch model
#     x: input tensor (requires_grad=True)
#     y_true: target class (integer or one-hot)
#     epsilon: perturbation magnitude
#     """
#     x_adv = x.clone().detach().requires_grad_(True)
#     output = model(x_adv)
#     loss = F.cross_entropy(output, y_true)
#     loss.backward()
#     perturbation = epsilon * x_adv.grad.sign()
#     x_adv = x_adv + perturbation
#     return x_adv.detach()


# def estimate_probabilistic_robustness(
#     fit,
#     x_test_sample,
#     y_true,
#     sample_indices,
#     input_dim,
#     hidden_dim,
#     output_dim,
#     epsilon=0.1,
#     delta = 0.1,
#     p_norm = 1,
#     activation = F.relu
# ):
#     """
#     Estimate the probability that f_w(x) == f_w(x_adv) under the posterior.
    
#     Returns:
#         robustness_prob: Monte Carlo estimate
#         robustness_flags: list of bools per sample
#     """
#     #from utils.stan_to_torch import build_pytorch_model_from_stan_sample

#     p_1_flags = []
#     p_2_flags = []

#     x_test_tensor = torch.tensor(x_test_sample, dtype=torch.float32)

#     for idx in tqdm.tqdm(sample_indices, disable=True): # desc="Sampling models", ):
#         model = build_pytorch_model_from_stan_sample(
#             fit=fit,
#             sample_idx=idx,
#             input_dim=input_dim,
#             hidden_dim=hidden_dim,
#             output_dim=output_dim,
#             task="classification",
#             activation=activation
#         )
        
#         model.eval()
#         x_tensor = x_test_tensor.clone().detach().unsqueeze(0).requires_grad_(True)
#         y_tensor = torch.tensor([y_true])
#         # Generate adversarial example
#         x_adv = fgsm_attack(model, x_tensor, y_tensor, epsilon)
#         probs_orig = F.softmax(model(x_tensor), dim=1)
#         probs_adv = F.softmax(model(x_adv), dim=1)
        
#         shift = torch.norm(probs_orig - probs_adv, p=p_norm).item()
#         is_softmax_shift = shift > delta

        
#         label_orig = torch.argmax(probs_orig).item()
#         label_adv = torch.argmax(probs_adv).item()
#         is_label_changed = label_orig != label_adv

#         p_1_flags.append(is_softmax_shift)
#         p_2_flags.append(is_label_changed)

#     p_1_prob = np.mean(p_1_flags)
#     p_2_prob = np.mean(p_2_flags)
#     return p_1_prob, p_1_flags, p_2_prob, p_2_flags


# def estimate_robustness_for_all_models(
#     fits_dict,
#     x_test_sample,
#     y_true,
#     input_dim,
#     hidden_dim,
#     output_dim,
#     sample_indices,
#     epsilon=0.1,
#     delta=0.1,
#     p_norm=1,
#     activation = F.relu
# ):
#     """
#     Compute probabilistic robustness estimates (p₁ and p₂) for each model in a fit dictionary.

#     Parameters:
#         fits_dict: dict from model name to {'posterior': CmdStanMCMC}
#         x_test_sample: 1D NumPy array (single test input)
#         y_true: integer label (0-indexed)
#         sample_indices: which posterior samples to use
#         epsilon: FGSM epsilon for perturbation
#         delta: threshold for softmax shift (for p₁)
#         p_norm: norm used to measure softmax shift (typically 1 or 2)

#     Returns:
#         dict: model name → {'p1': float, 'p2': float}
#     """
#     results = {}

#     for model_name, model_entry in fits_dict.items():
#         fit = model_entry["posterior"]

#         p1_prob, _, p2_prob, _ = estimate_probabilistic_robustness(
#             fit=fit,
#             x_test_sample=x_test_sample,
#             y_true=y_true,
#             sample_indices=sample_indices,
#             input_dim=input_dim,
#             hidden_dim=hidden_dim,
#             output_dim=output_dim,
#             epsilon=epsilon,
#             delta=delta,
#             p_norm=p_norm,
#             activation=activation
#         )

#         results[model_name] = {
#             "p1": p1_prob,
#             "p2": p2_prob
#         }

#     return results


# def estimate_robustness_over_test_set(
#     x_test, 
#     y_test, 
#     fits_dict,
#     input_dim,
#     hidden_dim,
#     output_dim,
#     sample_indices,
#     epsilon=0.1,
#     delta=0.1,
#     p_norm=1,
#     activation=F.relu,
#     ):
#     all_results = []
#     for i in tqdm.tqdm(range(len(x_test)), desc="Test samples", disable=False):
#         x_i = x_test.iloc[i].to_numpy()
#         y_i = y_test.iloc[i] 
        
#         robustness_results = estimate_robustness_for_all_models(
#         fits_dict=fits_dict,
#         x_test_sample=x_i,
#         y_true=y_i,
#         input_dim=input_dim,
#         hidden_dim=hidden_dim,
#         output_dim=output_dim,
#         sample_indices=sample_indices,
#         epsilon=epsilon,
#         delta=delta,
#         activation=activation
#         )
        
#         for model, robustness in robustness_results.items():
#             all_results.append({
#                 "epsilon": epsilon,
#                 "model": model,
#                 "robustness": robustness
#             })
            
#     df_results = pd.DataFrame(all_results)
#     return df_results
