// =====================
// Prior predictive model with non-centered parameterization
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

    hidden[1] = fmax((X * W_1 + rep_vector(1.0, N) * hidden_bias[1]), 0);

    if (L > 1) {
      for (l in 2:L)
        hidden[l] = tanh(hidden[l - 1] * W_internal[l - 1] + rep_vector(1.0, N) * hidden_bias[l]);
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
  matrix[N, output_nodes] y;
  
  int<lower=1> L;
  int<lower=1> H;
  
  int<lower=1> N_test;
  matrix[N_test, P] X_test;
  
  int<lower=1> p_0;
  real<lower=0> a;
  real<lower=0> b;
  vector<lower=0>[P] alpha;
  //real<lower=0, upper=1> gamma;
}

parameters {
  vector<lower=0, upper=50>[H] lambda;
  array[H] simplex[P] phi_data;
  real<lower=1e-6> tau;
  vector<lower=0>[H] c_sq;

  matrix[P, H] W1_raw;

  array[max(L - 1, 1)] matrix[H, H] W_internal;
  array[L] row_vector[H] hidden_bias;
  matrix[H, output_nodes] W_L;
  row_vector[output_nodes] output_bias;
  real<lower=1e-6> sigma;
}

transformed parameters {
  real<lower=1e-6> tau_0 = (p_0 * 1.0) / (P - p_0) * 1 / sqrt(N);

  vector<lower=0>[H] lambda_tilde_data;

  for (j in 1:H) {
    lambda_tilde_data[j] = fmax(1e-12, c_sq[j] * square(lambda[j]) /
                              (c_sq[j] + square(lambda[j]) * square(tau)));
  }

  matrix[P, H] W_1;
  for (j in 1:H) {
    for (i in 1:P) {
      real stddev = fmax(1e-12, tau * sqrt(lambda_tilde_data[j]) * sqrt(phi_data[j][i]));
      W_1[i, j] = stddev * W1_raw[i, j];
    }
  }

  matrix[N, output_nodes] output = nn_predict(X, 
                                W_1, W_internal, hidden_bias, 
                                W_L, output_bias, L);
}

model {
  tau ~ cauchy(0, tau_0);
  c_sq ~ inv_gamma(a, b);

  for (j in 1:H) {
    lambda[j] ~ student_t(3, 0, 1);
    phi_data[j] ~ beta(alpha, (P-1)*alpha);
  }
  to_vector(W1_raw) ~ normal(0, 1);

  if (L > 1) {
    for (l in 1:(L - 1)) {
      for (j in 1:H) {
        W_internal[l][, j] ~ normal(0, 1);
      }
    }
  }

  for (l in 1:L)
    hidden_bias[l] ~ normal(0, 1);

  for (j in 1:output_nodes)
    W_L[, j] ~ normal(0, 1);

  output_bias ~ normal(0, 1);
  sigma ~ inv_gamma(3, 2);

  // Likelihood
  for (n in 1:N)
    for (j in 1:output_nodes)
      y[n, j] ~ normal(output[n, j], sigma);
}

generated quantities {
  matrix[N, output_nodes] output_dbg = output;
  matrix[N_test, output_nodes] output_test = nn_predict(
    X_test,
    W_1,
    W_internal,
    hidden_bias,
    W_L,
    output_bias,
    L
  );
  matrix[N_test, output_nodes] output_test_rng;
  for (n in 1:N_test)
    for (j in 1:output_nodes)
      output_test_rng[n, j] = normal_rng(output_test[n, j], sigma);
}