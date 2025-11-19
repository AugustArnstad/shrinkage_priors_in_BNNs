// =====================
// Bayesian Neural Network with PCP on W_1 (Gaussian output)
// =====================

functions {
  matrix nn_predict(matrix X,
                    matrix W_1,
                    array[] matrix W_internal,
                    array[] row_vector hidden_bias,
                    matrix W_L,
                    row_vector output_bias,
                    int L) {
    int N = rows(X);
    int output_nodes = cols(W_L);
    int H = cols(W_1);
    array[L] matrix[N, H] hidden;

    hidden[1] = tanh(X * W_1 + rep_vector(1.0, N) * hidden_bias[1]);

    if (L > 1) {
      for (l in 2:L)
        hidden[l] = tanh(hidden[l - 1] * W_internal[l - 1] + rep_vector(1.0, N) * hidden_bias[l]);
    }

    matrix[N, output_nodes] output = hidden[L] * W_L;
    output += rep_matrix(output_bias, N);
    return output;
  }

  // PCP: log-density of log-Cauchy(0,1) on kappa
  real log_lcauchy_kappa(real kappa) {
    // safeguard; caller ensures kappa>0
    return -log(pi()) - log(kappa) - log1p(square(log(kappa)));
  }
}

data {
  int<lower=1> N;
  int<lower=1> P;
  matrix[N, P] X;
  int<lower=1> output_nodes;
  array[N] int<lower=1, upper=output_nodes> y;

  int<lower=1> L;
  int<lower=1> H;

  int<lower=1> N_test;
  matrix[N_test, P] X_test;

  // PCP: tiny epsilon to ensure kappa > 0 numerically
  real<lower=0> kappa_eps; // e.g. 1e-12
}

transformed data {
  // PCP: sum over n of ||x_n||^2 (only depends on X)
  real sum_x_norm2 = 0;
  for (n in 1:N)
    sum_x_norm2 += dot_self( to_vector(X[n]) );
}

parameters {
  // PCP: W_1 scale parameter (subject of the PCP prior)
  real<lower=0> tau1;

  matrix[P, H] W_1;
  array[max(L - 1, 1)] matrix[H, H] W_internal;
  array[L] row_vector[H] hidden_bias;
  matrix[H, output_nodes] W_L;
  row_vector[output_nodes] output_bias;
  real<lower=1e-6> sigma;
}

transformed parameters {
  matrix[N, output_nodes] output = nn_predict(X, W_1, W_internal,
                                              hidden_bias, W_L,
                                              output_bias, L);
  // Optional: expose kappa & C for monitoring
  real frob_WL = 0;
  for (j in 1:output_nodes)
    frob_WL += dot_self( to_vector(W_L[, j]) );
  real C = (frob_WL / (2.0 * square(sigma) * H * N)) * sum_x_norm2; // PCP constant (L=1 linearization)
  real kappa = C * tau1 + kappa_eps; // PCP: mapping for the prior
}

model {
  // ---- Priors ----

  // PCP: W_1 prior uses tau1 as variance scale (per-column var = tau1/H)
  // (Centered form kept to minimize changes)
  to_vector(W_1) ~ normal(0, sqrt(tau1) / sqrt(H));

  // Internal layers (unchanged)
  if (L > 1) {
    for (l in 1:(L - 1))
      for (j in 1:H)
        W_internal[l][, j] ~ normal(0, 1);
  }

  for (l in 1:L)
    hidden_bias[l] ~ normal(0, 1);

  for (j in 1:output_nodes)
    W_L[, j] ~ normal(0, 1);

  output_bias ~ normal(0, 1);
  sigma ~ inv_gamma(3, 2);

  // PCP prior on tau1 via kappa = C * tau1 (both computed in transformed parameters)
  target += log_lcauchy_kappa(kappa);
  target += log(C);               // Jacobian term; if you want a safety jitter: log(C + 1e-12)


  // ---- Likelihood ----
  for (n in 1:N)
      y[n] ~ categorical_logit(to_vector(output[n]));
}

generated quantities {
  matrix[N, output_nodes] output_dbg = output;
  matrix[N_test, output_nodes] output_test = nn_predict(
    X_test, W_1, W_internal,
    hidden_bias, W_L, output_bias, L);

  matrix[N_test, output_nodes] prob_test;
  array[N_test] int pred_test;

  for (n in 1:N_test) {
    vector[output_nodes] probs = softmax(to_vector(output_test[n]));
    prob_test[n] = to_row_vector(probs);
    pred_test[n] = categorical_rng(probs);
  }

  vector[N] log_lik;
  for (n in 1:N)
    log_lik[n] = categorical_logit_lpmf(y[n] | to_vector(output[n]));
}

