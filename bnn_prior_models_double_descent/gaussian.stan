// =====================
// Bayesian Neural Network with Gaussian prior
// =====================


data {
  int<lower=1> N;
  int<lower=1> P;
  matrix[N, P] X;
  int<lower=1> output_nodes;
  //matrix[N, output_nodes] y;

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
  real<lower=1e-6> sigma;
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

}

