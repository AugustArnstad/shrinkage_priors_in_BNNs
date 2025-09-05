// =====================
// Bayesian Neural Network with Gaussian prior - Classification Version
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

    hidden[1] = fmax(X * W_1 + rep_vector(1.0, N) * hidden_bias[1], 0);

    if (L > 1) {
      for (l in 2:L)
        hidden[l] = fmax(hidden[l - 1] * W_internal[l - 1] + rep_vector(1.0, N) * hidden_bias[l], 0);
    }

    matrix[N, output_nodes] output = hidden[L] * W_L;
    output += rep_matrix(output_bias, N);
    return output;
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
}

parameters {
  matrix[P, H] W_1;
  array[max(L - 1, 1)] matrix[H, H] W_internal;
  array[L] row_vector[H] hidden_bias;
  matrix[H, output_nodes] W_L;
  row_vector[output_nodes] output_bias;
}

transformed parameters {
  matrix[N, output_nodes] output = nn_predict(X, W_1, W_internal,
                                              hidden_bias, W_L,
                                              output_bias, L);
}

model {
  to_vector(W_1) ~ normal(0, 1);

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