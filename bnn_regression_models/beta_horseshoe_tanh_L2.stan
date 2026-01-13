// =====================
// Prior predictive model with non-centered parameterization
// =====================

functions {
  matrix nn_predict(matrix X,
                    matrix W_1,
                    matrix W_2,
                    array[] row_vector hidden_bias,
                    matrix W_L,
                    row_vector output_bias,
                    int L) {
    int N = rows(X);
    int H = cols(W_1);
    int output_nodes = cols(W_L);
    array[L] matrix[N, H] hidden;

    hidden[1] = tanh(X * W_1 + rep_vector(1.0, N) * hidden_bias[1]);
    hidden[2] = tanh(hidden[1] * W_2 + rep_vector(1.0, N) * hidden_bias[2]);

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
  vector<lower=0>[H] alpha_h;
  real<lower=0, upper=1> gamma;
}

parameters {
  vector<lower=1e-6>[L] tau;

  array[H] vector<lower=0, upper=50>[P] lambda_data;
  array[H] simplex[P] phi_data;
  vector<lower=0>[H] c_sq;
  matrix[P, H] W1_raw;

  array[H] vector<lower=0, upper=50>[H] lambda2_data;
  array[H] simplex[H] phi2_data;
  vector<lower=0>[H] c2_sq;
  matrix[H, H] W2_raw;

  array[L] row_vector[H] hidden_bias;
  matrix[H, output_nodes] W_L;
  row_vector[output_nodes] output_bias;
  real<lower=1e-6> sigma;
}

transformed parameters {
  real<lower=1e-6> tau0_1 = (p_0 * 1.0) / (P - p_0) * inv_sqrt(N);
  real<lower=1e-6> tau0_2 = (p_0 * 1.0) / (H - p_0) * inv_sqrt(N);

  array[H] vector<lower=0>[P] lambda_tilde_data;
  for (j in 1:H) {
    for (i in 1:P) {
      lambda_tilde_data[j][i] = fmax(1e-12, c_sq[j] * square(lambda_data[j][i]) /
                                (c_sq[j] + square(lambda_data[j][i]) * square(tau[1])));
    }
  }

  matrix[P, H] W_1;
  for (j in 1:H) {
    for (i in 1:P) {
      real stddev = fmax(1e-12, tau[1] * sqrt(lambda_tilde_data[j][i]) * sqrt(phi_data[j][i]));
      W_1[i, j] = stddev * W1_raw[i, j];
    }
  }

  array[H] vector<lower=0>[H] lambda2_tilde_data;
  for (j in 1:H) {
    for (i in 1:H) {
      lambda2_tilde_data[j][i] =
        fmax(1e-12,
             c2_sq[j] * square(lambda2_data[j][i]) /
             (c2_sq[j] + square(lambda2_data[j][i]) * square(tau[2])));
    }
  }

  matrix[H, H] W_2;
  for (j in 1:H) {
    for (i in 1:H) {
      real stddev2 = fmax(1e-12,
                          tau[2] * sqrt(lambda2_tilde_data[j][i]) * sqrt(phi2_data[j][i]));
      W_2[i, j] = stddev2 * W2_raw[i, j];
    }
  }

  matrix[N, output_nodes] output = nn_predict(X, 
                                W_1, W_2, hidden_bias, 
                                W_L, output_bias, L);
}

model {
  tau[1] ~ cauchy(0, tau0_1);
  tau[2] ~ cauchy(0, tau0_2);
  c_sq ~ inv_gamma(a, b);

  for (j in 1:H) {
    lambda_data[j] ~ cauchy(0, 1);
    phi_data[j] ~ beta(alpha, (P-1)*alpha);
  }
  to_vector(W1_raw) ~ normal(0, 1);

  c2_sq ~ inv_gamma(a, b);

  for (j in 1:H) {
    lambda2_data[j] ~ cauchy(0, 1);
    phi2_data[j] ~ beta(alpha_h, (P-1)*alpha_h);
  }
  to_vector(W2_raw) ~ normal(0, 1);

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
    W_2,
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