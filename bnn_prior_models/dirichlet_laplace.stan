// =====================
// Prior predictive model with a single hidden layer (L = 1)
// =====================

functions {
  matrix nn_predict(matrix X,
                    matrix data_to_hidden_mat,
                    array[] matrix hidden_to_hidden,
                    array[] row_vector hidden_biases,
                    matrix hidden_to_output,
                    row_vector output_bias,
                    int L) {
    int N = rows(X);
    int output_nodes = cols(hidden_to_output);
    int H = cols(data_to_hidden_mat);
    array[L] matrix[N, H] hidden;

    hidden[1] = fmax((X * data_to_hidden_mat + rep_vector(1.0, N) * hidden_biases[1]), 0);

    if (L > 1) {
      for (l in 2:L)
        hidden[l] = fmax((hidden[l - 1] * hidden_to_hidden[l - 1] + rep_vector(1.0, N) * hidden_biases[l]), 0);
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

  vector<lower=0>[P] alpha;
}

parameters {
  array[H] simplex[P] phi;
  real<lower=0> tau;
  array[H] vector<lower=0>[P] psi;

  array[H] vector[P] z_data_to_hidden;
  array[max(L - 1, 1)] matrix[H, H] hidden_to_hidden;
  array[L] row_vector[H] hidden_biases;
  matrix[H, output_nodes] hidden_to_output;
  row_vector[output_nodes] output_bias;
  real<lower=1e-6> sigma;
}

transformed parameters {
  real alpha_value = alpha[1];

  array[H] vector<lower=0>[P] phi_tilde;
  for (j in 1:H) {
    for (i in 1:P) {
      phi_tilde[j][i] = P * phi[j][i];
    }
  }

  array[H] vector[P] data_to_hidden;
  for (j in 1:H) {
    for (i in 1:P) {
      //psi[j][i] ~ exponential(1 / (2*square(phi_tilde[j][i] * tau)));
      real stddev = fmax(1e-12, sqrt(sqrt(psi[j][i])));
      data_to_hidden[j][i] = stddev * z_data_to_hidden[j][i];
    }
  }

  matrix[P, H] data_to_hidden_mat;
  for (j in 1:H) {
    for (i in 1:P) {
      data_to_hidden_mat[i, j] = data_to_hidden[j][i];
    }
  }

  matrix[N, output_nodes] output = nn_predict(X, 
                                data_to_hidden_mat, hidden_to_hidden, hidden_biases, 
                                hidden_to_output, output_bias, L);
}

model {
  // Prior
  for (j in 1:H) {
    phi[j] ~ dirichlet(alpha);
  }
  tau ~ gamma(P * alpha_value, 0.5);

  for (j in 1:H) {
    for (i in 1:P) {
      psi[j][i] ~ exponential(1 / (2*square(phi_tilde[j][i] * tau)));
    }
  }

  for (j in 1:H) {
    z_data_to_hidden[j] ~ normal(0, 1);
  }


  if (L > 1) {
    for (l in 1:(L - 1)) {
      for (j in 1:H) {
        hidden_to_hidden[l][, j] ~ normal(0, 1);
      }
    }
  }

  for (l in 1:L)
    hidden_biases[l] ~ normal(0, 1);

  for (j in 1:output_nodes)
    hidden_to_output[, j] ~ normal(0, 1);

  output_bias ~ normal(0, 1);
  sigma ~ inv_gamma(3, 2); //normal(0, 1);

  // Likelihood
  for (n in 1:N)
    for (j in 1:output_nodes)
      y[n, j] ~ normal(output[n, j], sigma);
}

generated quantities {
  matrix[N, output_nodes] output_dbg = output;
  matrix[N_test, output_nodes] output_test = nn_predict(
    X_test,
    data_to_hidden_mat,
    hidden_to_hidden,
    hidden_biases,
    hidden_to_output,
    output_bias,
    L
  );
  matrix[N_test, output_nodes] output_test_rng;
  for (n in 1:N_test)
    for (j in 1:output_nodes)
      output_test_rng[n, j] = normal_rng(output_test[n, j], sigma);

  vector[N] log_lik = rep_vector(0.0, N);
  for (n in 1:N)
    for (j in 1:output_nodes)
      log_lik[n] += normal_lpdf(y[n, j] | output[n, j], sigma);
}
