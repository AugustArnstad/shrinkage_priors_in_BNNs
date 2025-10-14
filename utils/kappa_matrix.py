import numpy as np
from typing import Tuple, Callable

# ---------- Aktivasjon og deriverte ----------

def get_activation(activation: str = "tanh") -> Tuple[Callable, Callable]:
    if activation == "tanh":
        phi = np.tanh
        def dphi(a): return 1.0 - np.tanh(a)**2
    elif activation == "relu":
        def phi(a): return np.maximum(0.0, a)
        def dphi(a): return (a > 0.0).astype(a.dtype)
    else:
        raise ValueError(f"Unsupported activation: {activation}")
    return phi, dphi

# ---------- H(w0) og J_W(w0, v0) ----------

def build_hidden_and_jacobian_W(
    X: np.ndarray,               # (n, p)
    W0: np.ndarray,              # (H, p)  -- vekter i referansepunktet w0
    b0: np.ndarray,              # (H,)    -- bias i referansepunktet w0
    v0: np.ndarray,              # (H,)    -- utgangsvekter i referansepunktet v0
    activation: str = "tanh",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returnerer:
      H  : (n, H)         = H(w0)
      JW : (n, H*p)       = d(H(w)v)/d vec(W) |_(w0, v0), kolonner ordnet som (h=0..H-1, j=0..p-1)
    """
    n, p = X.shape
    H, pW = W0.shape
    assert pW == p
    phi, dphi = get_activation(activation)

    # Pre- og post-aktivert
    A = X @ W0.T + b0[None, :]        # (n, H), a_{i,h}
    Phi_mat = phi(A)                     # (n, H), h_{i,h}
    dphiA = dphi(A)                   # (n, H)
    
    # J_W: df/dW_{h,j} = v_h * dphi(a_{i,h}) * x_{i,j}
    # For hver node h bygger vi et (n, p)-bidrag og flater ut langs j, og stabler så langs h.
    JW_blocks = []
    for h in range(H):
        # (n,1) * (1,p) -> (n,p)
        block_h = (v0[h] * dphiA[:, [h]]) * X 
        JW_blocks.append(block_h.reshape(n, p))
    # Stack kolonnevis i rekkefølge (h, j) -> (n, H*p)
    Jb = dphiA * v0[None, :]     # (n, H)
    JW = np.hstack([B for B in JW_blocks])
    Joutb = np.ones(n)        # (n,)
    return Phi_mat, JW, Jb, Joutb

# ---------- Sigma_y og P ----------

def build_Sigma_y(
    Phi_mat: np.ndarray,   # (n, H) = H(w0)
    tau_v: float,          # prior std for v
    noise: float,          # likelihood std
    J_b1: np.ndarray = None,     # (n, H), optional
    J_b2: np.ndarray = None,     # (n,),   optional
    include_b1: bool = True,
    include_b2: bool = True,
) -> np.ndarray:
    """
    Σ_y = τ_v^2 ΦΦ^T + [J_b1 J_b1^T if include_b1] + [J_b2 J_b2^T if include_b2] + σ^2 I_n
    """
    n = Phi_mat.shape[0]
    Sigma_y = (tau_v**2) * (Phi_mat @ Phi_mat.T) + (noise**2) * np.eye(n)

    if include_b1 and (J_b1 is not None):
        Sigma_y = Sigma_y + (J_b1 @ J_b1.T)

    if include_b2 and (J_b2 is not None):
        Sigma_y = Sigma_y + np.outer(J_b2, J_b2)

    return Sigma_y


def build_P_from_lambda_tau(
    lambda_tilde: np.ndarray,  # (H, p) lokale skalaer for W
    tau_w: float               # global skala for w
) -> np.ndarray:
    """
    P = τ_w^{-2} Λ^{-1} der Λ = diag(λ^2) for konsistens med uttrykket 1/(1 + τ^2 λ^2 s).
    Dvs. diag(P) = 1 / (τ_w^2 * λ^2).
    Returnerer P som (H*p, H*p) diagonalmatrise.
    """
    lam_vec = lambda_tilde.reshape(-1)          # (H*p,)
    diagP = 1.0 / ( (tau_w**2) * (lam_vec) ) # (H*p,)
    return np.diag(diagP)

# ---------- S, shrinkage-matrise R = (P+S)^{-1} P ----------

def build_S(JW: np.ndarray, Sigma_y: np.ndarray) -> np.ndarray:
    """
    S = J_W^T Σ_y^{-1} J_W  (Hp x Hp).
    Løser via lineær solve for stabilitet: X = Σ_y^{-1} J_W = solve(Σ_y, J_W).
    """
    X = np.linalg.solve(Sigma_y, JW)       # (n, Hp)
    return JW.T @ X                        # (Hp, Hp)

def shrinkage_matrix(P: np.ndarray, S: np.ndarray) -> np.ndarray:
    """
    R = (P+S)^{-1} P. Bruk Cholesky når mulig.
    Løser (P+S) * R = P for R.
    """
    A = P + S
    # Robust fallback hvis Cholesky feiler
    try:
        L = np.linalg.cholesky(A)
        # L Y = P  -> Y
        Y = np.linalg.solve(L, P)
        # L^T R = Y -> R
        R = np.linalg.solve(L.T, Y)
    except np.linalg.LinAlgError:
        R = np.linalg.solve(A, P)
    return R

def shrinkage_matrix_stable(P, S, jitter=0.0):
    """
    Stabil beregning av R = (P+S)^{-1} P via
    R = P^{1/2} (I + P^{-1/2} S P^{-1/2})^{-1} P^{1/2}.
    Krever at P er diagonal (positiv).
    """
    d = np.diag(P).astype(float)
    # Guardrails: ingen nuller/NaN/negativ
    eps = 1e-12
    d = np.clip(d, eps, np.finfo(float).max)
    Phalf    = np.diag(np.sqrt(d))
    Pinvhalf = np.diag(1.0 / np.sqrt(d))

    M = Pinvhalf @ S @ Pinvhalf
    # Jitter for SPD-sikkerhet (skader ikke i praksis)
    if jitter > 0:
        M = M + jitter * np.eye(M.shape[0])

    # (I + M) er SPD -> Cholesky
    I = np.eye(M.shape[0])
    L = np.linalg.cholesky(I + M)
    # (I+M)^{-1} P^{1/2} = (L^T)^{-1} (L)^{-1} P^{1/2}
    Z = np.linalg.solve(L, Phalf)
    W = np.linalg.solve(L.T, Z)
    # R = P^{1/2} * W
    #R = Phalf @ W
    R = Pinvhalf @ W
    # Symmetrer (numerisk)
    R = 0.5 * (R + R.T)
    return R

def shrinkage_eigs_and_df(P, S):
    """Returner r-eigenverdier og df_eff i P-whitnede koordinater."""
    d = np.diag(P).astype(float)
    eps = 1e-12
    Pinvhalf = np.diag(1.0 / np.sqrt(np.maximum(d, eps)))

    M = Pinvhalf @ S @ Pinvhalf          # SPD
    mu = np.linalg.eigvalsh(M)           # >= 0
    r = 1.0 / (1.0 + mu)                 # i (0,1]
    df_eff = np.sum(1.0 - r)             # = sum mu/(1+mu) >= 0
    return r, df_eff

def extract_model_draws(
    fit_dict,
    model: str,
    *,
    lambda_effective_candidates = ("lambda_tilde", "lambda_tilde_data"),
    lambda_raw_candidates       = ("lambda", "lambda_data"),
    include_phi_for_dirichlet: bool = True,
    phi_name: str = "phi_data",
    lambda_kind: str = "effective",
):
    """
    Returns draws with a flexible way to pick lambda:

      W_all      : (D, H, p)
      b_all      : (D, H)
      v_all      : (D, H)
      c_all      : (D,)
      sigma_all  : (D,)
      tau_w_all  : (D,)
      tau_v_all  : (D,)   (ones if not present)
      lambda_all : (D, H, p)       <-- chosen lambda (effective/raw) per `lambda_kind`
      [lambda_raw_all]             <-- ONLY if lambda_kind == 'both'

    Conventions:
    - 'effective' lambda = the *regularized* local factor actually used in the weight std/var
      (e.g., `lambda_tilde` or `lambda_tilde_data`), optionally multiplied by `phi_data`
      for Dirichlet-type models if `include_phi_for_dirichlet=True`.
    - 'raw' lambda = the *unregularized* half-Cauchy parameter (e.g., `lambda` or `lambda_data`).

    Notes:
    - If the model looks Gaussian (by name or because 'tau' is absent), lambdas default to ones.
    - Shapes are coerced to (D,H,p) when possible; transposes are handled automatically.
    """

    post = fit_dict[model]['posterior']

    def _stan_var_or_none(name):
        try:
            return np.asarray(post.stan_variable(name))
        except Exception:
            return None

    def _coerce_DHp(arr, D, H, p):
        """Coerce Stan draws to shape (D,H,p). Accepts (D,H,p) or (D,p,H) or (D,p,H,1)/(D,1,H,p)."""
        if arr is None:
            return None
        shp = arr.shape
        if shp == (D, H, p):
            return arr
        if shp == (D, p, H):
            return np.transpose(arr, (0, 2, 1))
        # Common fallbacks (rare):
        if len(shp) == 4 and shp[0] == D:
            # drop singleton dims and retry
            squeezed = np.squeeze(arr)
            return _coerce_DHp(squeezed, D, H, p)
        raise ValueError(f"Cannot coerce lambda/phi array of shape {shp} to (D,{H},{p}).")

    # === Core weights/bias/sigma ===
    # W_1: (D, p, H) -> (D, H, p)
    W_1 = _stan_var_or_none("W_1")
    if W_1 is None:
        raise ValueError("Missing 'W_1' in posterior.")
    W_all = np.transpose(W_1, (0, 2, 1))
    D, H, p = W_all.shape

    # W_L: (D, H, out_nodes=1) -> (D, H)
    W_L = _stan_var_or_none("W_L")
    if W_L is None:
        raise ValueError("Missing 'W_L' in posterior.")
    v_all = W_L.reshape(D, -1)

    # hidden_bias: (D, 1, H) -> (D, H)
    b_1 = _stan_var_or_none("hidden_bias")
    if b_1 is None:
        raise ValueError("Missing 'hidden_bias' in posterior.")
    b_all = b_1.reshape(D, -1)

    # output_bias: (D, 1) -> (D,)
    b_2 = _stan_var_or_none("output_bias")
    if b_2 is None:
        raise ValueError("Missing 'output_bias' in posterior.")
    c_all = b_2.reshape(D)

    # sigma
    sigma_all = _stan_var_or_none("sigma")
    if sigma_all is None:
        raise ValueError("Missing 'sigma' in posterior.")
    sigma_all = sigma_all.reshape(D)

    # Detect model types (best-effort)
    is_gauss     = ("Gaussian" in model) or (_stan_var_or_none("tau") is None)
    is_dirichlet = ("Dirichlet" in model) or ("DST" in model)
    is_rhs       = ("Regularized Horseshoe" in model)

    # tau_w / tau_v
    if is_gauss:
        tau_w_all = np.ones(D)
        tau_v_all = np.ones(D)
    else:
        tau_w = _stan_var_or_none("tau")
        if tau_w is None:
            # Fallback if naming differs
            tau_w = _stan_var_or_none("tau_w")
        tau_w_all = tau_w.reshape(D)

        tau_v = _stan_var_or_none("tau_v")
        tau_v_all = np.ones(D) if tau_v is None else tau_v.reshape(D)

    # === Lambda extraction ===
    # (1) Effective lambda (regularized)
    lam_eff = None
    if not is_gauss:
        for nm in lambda_effective_candidates:
            arr = _stan_var_or_none(nm)
            if arr is not None:
                lam_eff = _coerce_DHp(arr, D, H, p)
                break

    # (2) Raw lambda (half-Cauchy)
    lam_raw = None
    if not is_gauss:
        for nm in lambda_raw_candidates:
            arr = _stan_var_or_none(nm)
            if arr is not None:
                lam_raw = _coerce_DHp(arr, D, H, p)
                break

    # (3) Optional Dirichlet multiplier phi_data
    phi_hp = None
    if include_phi_for_dirichlet and is_dirichlet:
        phi_arr = _stan_var_or_none(phi_name)
        if phi_arr is not None:
            phi_hp = _coerce_DHp(phi_arr, D, H, p)

    # Build the "lambda_all" to return per lambda_kind
    ones_DHp = np.ones((D, H, p))

    def _with_phi(lam):
        if lam is None:
            return None
        return lam * (phi_hp if phi_hp is not None else 1.0)

    # Defaults for Gaussian or missing variables
    lambda_eff_all = None
    lambda_raw_all = None

    if is_gauss:
        lambda_eff_all = ones_DHp
        lambda_raw_all = ones_DHp
    else:
        # effective
        if lam_eff is not None:
            lambda_eff_all = _with_phi(lam_eff) if is_dirichlet else lam_eff
        else:
            # if effective is missing, fall back gracefully
            lambda_eff_all = _with_phi(lam_raw) if (is_dirichlet and lam_raw is not None) else (lam_raw if lam_raw is not None else ones_DHp)

        # raw
        if lam_raw is not None:
            lambda_raw_all = lam_raw
        else:
            # if raw not present, fall back to effective or ones
            lambda_raw_all = lam_eff if lam_eff is not None else ones_DHp

    # === Return ===
    if lambda_kind == "effective":
        return W_all, b_all, v_all, c_all, sigma_all, tau_w_all, tau_v_all, lambda_eff_all
    elif lambda_kind == "raw":
        return W_all, b_all, v_all, c_all, sigma_all, tau_w_all, tau_v_all, lambda_raw_all
    elif lambda_kind == "both":
        # returns 9 items (adds lambda_raw_all at the end)
        return W_all, b_all, v_all, c_all, sigma_all, tau_w_all, tau_v_all, lambda_eff_all, lambda_raw_all
    else:
        raise ValueError("lambda_kind must be one of {'effective','raw','both'}.")

# ------- Knyt alt sammen -------

def compute_shrinkage_for_W_block(
    X: np.ndarray,
    W0: np.ndarray, b0: np.ndarray, v0: np.ndarray,
    noise: float, tau_w: float, tau_v: float,
    lambda_tilde: np.ndarray,
    activation: str = "tanh",
    include_b1_in_Sigma: bool = True,
    include_b2_in_Sigma: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returnerer (R, P, S, Sigma_y) der R = (P+S)^{-1} P for W-blokken.
    """
    Phi_mat, JW, Jb1, Jb2 = build_hidden_and_jacobian_W(X, W0, b0, v0, activation=activation)  # (n,H), (n,Hp)
    #Sigma_y = build_Sigma_y(Phi_mat, tau_v=tau_v, J_b1=Jb1, J_b2=Jb2, noise=noise)                       # (n,n)
    Sigma_y = build_Sigma_y(
        Phi_mat,
        tau_v=tau_v,
        noise=noise,
        J_b1=Jb1,
        J_b2=Jb2,
        include_b1=include_b1_in_Sigma,
        include_b2=include_b2_in_Sigma,
    )
    P = build_P_from_lambda_tau(lambda_tilde, tau_w=tau_w)                        # (Hp,Hp)
    S = build_S(JW, Sigma_y)                                                      # (Hp,Hp)
    R = shrinkage_matrix_stable(P, S)                                                    # (Hp,Hp)
    return R, P, S, Sigma_y, JW, Phi_mat

def compute_shrinkage(
    X,
    W_all, b_all, v_all,          # (D,H,p), (D,H), (D,H)
    sigma_all, tau_w_all, tau_v_all,  # (D,), (D,), (D,)
    lambda_all,                   # (D,H,p)
    activation="tanh",
    return_mats=True,             # set False if you only want summaries
    include_b1_in_Sigma: bool = True,
    include_b2_in_Sigma: bool = True,
):
    """
    Loop over draws and compute R=(P+S)^{-1}P per draw using your single-draw function.
    Returns:
      R_stack : (D, N, N) with N=H*p  (if return_mats=True, else None)
      r_eigs  : (D, N)  sorted eigenvalues in [0,1]
      df_eff  : (D,)    effective dof = tr(I-R) = N - tr(R)
    """
    D, H, p = W_all.shape
    N = H * p

    R_stack = np.empty((D, N, N)) if return_mats else None
    S_stack = np.empty((D, N, N)) if return_mats else None
    P_stack = np.empty((D, N, N)) if return_mats else None
    G_stack = np.empty((D, N, N)) if return_mats else None
    shrink_stack= np.empty((D, N, N)) if return_mats else None
    r_eigs  = np.empty((D, N))
    df_eff  = np.empty(D)

    for d in range(D):
        R, P, S, Sigma_y, _, _ = compute_shrinkage_for_W_block(
            X=X,
            W0=W_all[d],
            b0=b_all[d],
            v0=v_all[d],
            noise=float(sigma_all[d]),
            tau_w=float(tau_w_all[d]),
            tau_v=float(tau_v_all[d]),
            lambda_tilde=lambda_all[d],
            activation=activation,
            include_b1_in_Sigma=include_b1_in_Sigma,
            include_b2_in_Sigma=include_b2_in_Sigma,
        )
        p = np.diag(P)                       
        P_inv_sqrt = np.diag(1.0/np.sqrt(p))         
        G = P_inv_sqrt @ S @ P_inv_sqrt 
        I = np.identity(N)
        shrink_mat = np.linalg.inv(I + G)@G

        if return_mats:
            R_stack[d] = R
            S_stack[d] = S
            P_stack[d] = P
            G_stack[d] = G
            shrink_stack[d] = shrink_mat
        


        r, df = shrinkage_eigs_and_df(P, S)
        r_eigs[d] = np.sort(r)
        df_eff[d] = df

    return R_stack, S_stack, P_stack, G_stack, shrink_stack, r_eigs, df_eff

### PLOT UTILITIES:


import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm, Normalize, LinearSegmentedColormap

def clamp_small(x, tol=1e-9):
    return 0.0 if abs(x) < tol else x

def add_block_grid(ax, H, p, color="w", lw=0.5):
    Hp = H*p
    for h in range(1, H):
        k = h*p
        ax.axhline(k-0.5, color=color, lw=lw)
        ax.axvline(k-0.5, color=color, lw=lw)

def asymmetric_diverging_cmap(vmin, vmax, neg='#2b6cb0', pos='#d53e4f', eps=1e-12):
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = -1.0, 1.0
    p = float(np.clip((0.0 - vmin) / (vmax - vmin), 0.0, 1.0))
    # Ensure black at the end if 0 coincides with vmin or vmax
    if p <= eps:
        colors = [(0.0, 'black'), (1.0, pos)]
    elif p >= 1.0 - eps:
        colors = [(0.0, neg), (1.0, 'black')]
    else:
        colors = [(0.0, neg), (p, 'black'), (1.0, pos)]
    return LinearSegmentedColormap.from_list('asym_neg_black_pos', colors, N=256)

def visualize_models(
    matrices, names, H=16, p=10, use_abs=False, q_low=0.05, q_high=0.99,
    neg_color='#2b6cb0', pos_color='#d53e4f'
):
    """
    Per-model color scale:
      - vmin = q_low quantile of values in that matrix
      - vmax = q_high quantile
      - 0 maps to black, and sits at the correct fractional position between vmin and vmax.
    """
    mats = [np.abs(M) if use_abs else M for M in matrices]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=150, constrained_layout=True)
    axes = axes.ravel()

    for ax, M, title in zip(axes, mats, names):
        vals = M[np.isfinite(M)]
        if vals.size == 0:
            vmin, vmax = -1.0, 1.0
        else:
            vmin = float(np.quantile(vals, q_low))
            vmax = float(np.quantile(vals, q_high))
            # ensure 0 is inside range so black appears; nudge if necessary
            eps = 1e-12
            if vmin >= 0: vmin = -eps
            if vmax <= 0: vmax =  eps

        # Etter at du har beregnet vmin, vmax:
        vmin = clamp_small(vmin, tol=1e-9)
        vmax = clamp_small(vmax, tol=1e-9)

        # Sikre at området ikke kollapser:
        if abs(vmax - vmin) < 1e-12:
            vmin, vmax = -1e-9, 1e-9
        # Build an asymmetric diverging cmap for THIS panel, then use linear Normalize
        cmap = asymmetric_diverging_cmap(vmin, vmax, neg=neg_color, pos=pos_color)
        norm = PowerNorm(gamma=0.5, vmin=vmin, vmax=vmax)   # gamma<1 boosts mid values; >1 compresses
        #norm = Normalize(vmin=vmin, vmax=vmax)

        im = ax.imshow(M, aspect='equal', interpolation='nearest', cmap=cmap, norm=norm)

        # Optional: draw your block grid if you have this helper defined elsewhere
        try:
            add_block_grid(ax, H, p)
        except NameError:
            pass

        ax.set_title(title)
        ax.set_xlabel("Columns"); ax.set_ylabel("Rows")

        cb = fig.colorbar(im, ax=ax, shrink=1.0)#, pad=0.02)
        cb.set_label("Value")

        # Put a tick at zero explicitly
        ticks = np.linspace(vmin, vmax, 5)
        ticks[2] = 0.0
        ticks.sort()
        cb.set_ticks(ticks)

        print(f"{title}: vmin (q={q_low}) = {vmin:.2g}, vmax (q={q_high}) = {vmax:.2g}")

    plt.show()

def _symmetrize(M):
    return 0.5*(M + M.swapaxes(-1, -2))

def _spd_log(M, eps=1e-8):
    M = _symmetrize(M)
    w, U = np.linalg.eigh(M + eps*np.eye(M.shape[-1]))
    return U @ np.diag(np.log(np.clip(w, eps, None))) @ U.T

def _spd_exp(M):
    M = _symmetrize(M)
    w, U = np.linalg.eigh(M)
    return U @ np.diag(np.exp(w)) @ U.T

def log_euclidean_median(stack, eps=1e-8):
    """
    stack: (D, N, N), SPD/PSD. Uses elementwise median in log-domain.
    Returns: (N, N) SPD.
    """
    D, N, _ = stack.shape
    logs = np.empty_like(stack)
    for d in range(D):
        logs[d] = _spd_log(stack[d], eps=eps)
    med_log = np.median(logs, axis=0)
    return _spd_exp(med_log)


