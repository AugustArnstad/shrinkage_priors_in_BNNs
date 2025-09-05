// NOTE: This is not in use

functions {
  matrix nn_predict(matrix X,
                    matrix data_to_hidden,
                    array[] matrix hidden_to_hidden,
                    array[] row_vector hidden_biases,
                    matrix hidden_to_output,
                    row_vector output_bias,
                    int L) {
    int N = rows(X);
    int output_nodes = cols(hidden_to_output);
    int H = cols(data_to_hidden);
    array[L] matrix[N, H] hidden;

    hidden[1] = fmax(X * data_to_hidden + rep_vector(1.0, N) * hidden_biases[1], 0);

    if (L > 1) {
      for (l in 2:L)
        hidden[l] = fmax(hidden[l - 1] * hidden_to_hidden[l - 1] + rep_vector(1.0, N) * hidden_biases[l], 0);
    }

    matrix[N, output_nodes] output = hidden[L] * hidden_to_output;
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

  vector<lower=0>[P] lambda;  // Lasso shrinkage (fixed)
}

parameters {
  matrix<lower=1e-6, upper=10>[P, H] alpha;
  matrix[P, H] data_to_hidden_raw;  

  // matrix[P, H] data_to_hidden;
  array[max(L - 1, 1)] matrix[H, H] hidden_to_hidden;
  array[L] row_vector[H] hidden_biases;
  matrix[H, output_nodes] hidden_to_output;
  row_vector[output_nodes] output_bias;
  real<lower=1e-6> sigma;
  //real<lower=1e-6> sigma_hidden;
}

transformed parameters {
  matrix[P, H] data_to_hidden = alpha .* data_to_hidden_raw;

  matrix[N, output_nodes] output = nn_predict(X, data_to_hidden, hidden_to_hidden,
                                              hidden_biases, hidden_to_output,
                                              output_bias, L);
  
}

model {
  // Lasso prior on input-to-hidden weights

  to_vector(data_to_hidden_raw) ~ normal(0, 1);
  for (i in 1:P){
    for (j in 1:H){
      alpha[i, j] ~ exponential(1 / lambda[i]);
    }
  }

  // for (j in 1:H) {
  // data_to_hidden[, j] ~ double_exponential(rep_vector(0, P), lambda);
  // }

  // Rest: weakly regularized
  if (L > 1) {
    for (l in 1:(L - 1))
      for (j in 1:H)
        hidden_to_hidden[l][, j] ~ normal(0, 1);
  }
  
  for (l in 1:L)
    hidden_biases[l] ~ normal(0, 1);

  for (j in 1:output_nodes)
    hidden_to_output[, j] ~ normal(0, 1);

  output_bias ~ normal(0, 1);
  sigma ~ inv_gamma(1, 1);

  // Likelihood
  for (n in 1:N)
    for (j in 1:output_nodes)
      y[n, j] ~ normal(output[n, j], sigma);
}

generated quantities {
  matrix[N, output_nodes] output_dbg = output;
  matrix[N_test, output_nodes] output_test = nn_predict(X_test,
    data_to_hidden, hidden_to_hidden, hidden_biases,
    hidden_to_output, output_bias, L);

  matrix[N_test, output_nodes] output_test_rng;
  for (n in 1:N_test)
    for (j in 1:output_nodes)
      output_test_rng[n, j] = normal_rng(output_test[n, j], sigma);

  vector[N] log_lik = rep_vector(0.0, N);
  for (n in 1:N)
    for (j in 1:output_nodes)
      log_lik[n] += normal_lpdf(y[n, j] | output[n, j], sigma);
}