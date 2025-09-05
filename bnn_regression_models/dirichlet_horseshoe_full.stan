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
        hidden[l] = fmax((hidden[l - 1] * W_internal[l - 1] + rep_vector(1.0, N) * hidden_bias[l]), 0);
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
  vector<lower=0>[H] alpha_internal;
  real<lower=0, upper=1> gamma;
}

parameters {
  array[H] vector<lower=0, upper=50>[P] lambda_data;
  array[H] simplex[P] phi_data;
  real<lower=1e-6> tau;
  vector<lower=0>[H] c_sq;

  matrix[P, H] W1_raw;

  array[max(L - 1, 1)] matrix[H, H] W_internal_raw;
  
  array[H] vector<lower=0, upper=50>[H] lambda_2;
  array[H] simplex[H] phi_2;

  array[H] vector<lower=0, upper=50>[H] lambda_3;
  array[H] simplex[H] phi_3;

  array[H] vector<lower=0, upper=50>[H] lambda_4;
  array[H] simplex[H] phi_4;

  array[H] vector<lower=0, upper=50>[H] lambda_5;
  array[H] simplex[H] phi_5;

  array[H] vector<lower=0, upper=50>[H] lambda_6;
  array[H] simplex[H] phi_6;

  //array[H] vector<lower=0, upper=50>[H] lambda_7;
  //array[H] simplex[H] phi_7;

  array[max(L - 1, 1)] vector<lower=0>[H] c_sq_internal;

  array[L] row_vector[H] hidden_bias;
  //array[L] row_vector[H] bias;
  matrix[H, output_nodes] W_L;
  row_vector[output_nodes] output_bias;
  real<lower=1e-6> sigma;
}

transformed parameters {
  real<lower=1e-6> tau_0 = (p_0 * 1.0) / (P - p_0) * 1 / sqrt(N);

  array[H] vector<lower=0>[P] lambda_tilde_data;
  array[H] vector<lower=0>[P] phi_tilde_data;

  for (j in 1:H) {
    for (i in 1:P) {
      lambda_tilde_data[j][i] = fmax(1e-12, c_sq[j] * square(lambda_data[j][i]) /
                                (c_sq[j] + square(lambda_data[j][i]) * square(tau)));
      phi_tilde_data[j][i] = P * phi_data[j][i];
    }
  }

  matrix[P, H] W_1;
  for (j in 1:H) {
    for (i in 1:P) {
      real stddev = fmax(1e-12, tau * sqrt(lambda_tilde_data[j][i])) * phi_tilde_data[j][i];
      W_1[i, j] = stddev * W1_raw[i, j];
    }
  }

  array[H] vector<lower=0>[H] lambda_tilde_2;
  array[H] vector<lower=0>[H] phi_tilde_2;

  for (j in 1:H) {
    for (i in 1:H) {
      lambda_tilde_2[j][i] = fmax(1e-12, c_sq_internal[1][j] * square(lambda_2[j][i]) /
                                (c_sq_internal[1][j] + square(lambda_2[j][i]) * square(tau)));
      phi_tilde_2[j][i] = phi_2[j][i];
    }
  }

  array[max(L - 1, 1)] matrix[H, H] W_internal;

  for (j in 1:H) {
    for (i in 1:H) {
      real stddev = fmax(1e-12, tau * sqrt(lambda_tilde_2[j][i])) * phi_tilde_2[j][i];
      W_internal[1][i, j] = stddev * W_internal_raw[1][i, j];
    }
  }

  array[H] vector<lower=0>[H] lambda_tilde_3;
  array[H] vector<lower=0>[H] phi_tilde_3;

  for (j in 1:H) {
    for (i in 1:H) {
      lambda_tilde_3[j][i] = fmax(1e-12, c_sq_internal[2][j] * square(lambda_3[j][i]) /
                                (c_sq_internal[2][j] + square(lambda_3[j][i]) * square(tau)));
      phi_tilde_3[j][i] = phi_3[j][i];
    }
  }

  for (j in 1:H) {
    for (i in 1:H) {
      real stddev = fmax(1e-12, tau * sqrt(lambda_tilde_3[j][i])) * phi_tilde_3[j][i];
      W_internal[2][i, j] = stddev * W_internal_raw[2][i, j];
    }
  }

  array[H] vector<lower=0>[H] lambda_tilde_4;
  array[H] vector<lower=0>[H] phi_tilde_4;

  for (j in 1:H) {
    for (i in 1:H) {
      lambda_tilde_4[j][i] = fmax(1e-12, c_sq_internal[3][j] * square(lambda_4[j][i]) /
                                (c_sq_internal[3][j] + square(lambda_4[j][i]) * square(tau)));
      phi_tilde_4[j][i] = phi_4[j][i];
    }
  }

  for (j in 1:H) {
    for (i in 1:H) {
      real stddev = fmax(1e-12, tau * sqrt(lambda_tilde_4[j][i])) * phi_tilde_4[j][i];
      W_internal[3][i, j] = stddev * W_internal_raw[3][i, j];
    }
  }

  array[H] vector<lower=0>[H] lambda_tilde_5;
  array[H] vector<lower=0>[H] phi_tilde_5;

  for (j in 1:H) {
    for (i in 1:H) {
      lambda_tilde_5[j][i] = fmax(1e-12, c_sq_internal[4][j] * square(lambda_5[j][i]) /
                                (c_sq_internal[4][j] + square(lambda_5[j][i]) * square(tau)));
      phi_tilde_5[j][i] = phi_5[j][i];
    }
  }

  for (j in 1:H) {
    for (i in 1:H) {
      real stddev = fmax(1e-12, tau * sqrt(lambda_tilde_5[j][i])) * phi_tilde_5[j][i];
      W_internal[4][i, j] = stddev * W_internal_raw[4][i, j];
    }
  }

  array[H] vector<lower=0>[H] lambda_tilde_6;
  array[H] vector<lower=0>[H] phi_tilde_6;

  for (j in 1:H) {
    for (i in 1:H) {
      lambda_tilde_6[j][i] = fmax(1e-12, c_sq_internal[5][j] * square(lambda_6[j][i]) /
                                (c_sq_internal[5][j] + square(lambda_6[j][i]) * square(tau)));
      phi_tilde_6[j][i] = phi_6[j][i];
    }
  }

  for (j in 1:H) {
   for (i in 1:H) {
      real stddev = fmax(1e-12, tau * sqrt(lambda_tilde_6[j][i])) * phi_tilde_6[j][i];
      W_internal[5][i, j] = stddev * W_internal_raw[5][i, j];
    }
  }

  //array[H] vector<lower=0>[H] lambda_tilde_7;
  //array[H] vector<lower=0>[H] phi_tilde_7;

  //for (j in 1:H) {
  //  for (i in 1:H) {
  //    lambda_tilde_7[j][i] = fmax(1e-12, c_sq_internal[6][j] * square(lambda_7[j][i]) /
  //                              (c_sq_internal[6][j] + square(lambda_7[j][i]) * square(tau)));
  //    phi_tilde_7[j][i] = phi_7[j][i];
  //  }
  //}

  //for (j in 1:H) {
  //  for (i in 1:H) {
  //    real stddev = fmax(1e-12, tau * sqrt(lambda_tilde_7[j][i])) * phi_tilde_7[j][i];
  //    W_internal[6][i, j] = stddev * W_internal_raw[6][i, j];
  //  }
  //}

  matrix[N, output_nodes] output = nn_predict(X, 
                                W_1, W_internal, hidden_bias, 
                                W_L, output_bias, L);
  //matrix[N, output_nodes] output = nn_predict(X, 
  //                              W_1, hidden_bias, 
  //                              W_L, output_bias, L);
}

model {
  tau ~ cauchy(0, tau_0);
  c_sq ~ inv_gamma(a, b);

  for (j in 1:H) {
    lambda_data[j] ~ cauchy(0, 1);
    phi_data[j] ~ dirichlet(alpha);
  }
  to_vector(W1_raw) ~ normal(0, 1);

  for (l in 1:L - 1){
    c_sq_internal[l] ~ inv_gamma(a, b);
  }

  for (j in 1:H) {
    lambda_2[j] ~ cauchy(0, 1);
    phi_2[j] ~ dirichlet(alpha_internal);
  }
  for (j in 1:H) {
    lambda_3[j] ~ cauchy(0, 1);
    phi_3[j] ~ dirichlet(alpha_internal);
  }
  for (j in 1:H) {
    lambda_4[j] ~ cauchy(0, 1);
    phi_4[j] ~ dirichlet(alpha_internal);
  }
  for (j in 1:H) {
    lambda_5[j] ~ cauchy(0, 1);
    phi_5[j] ~ dirichlet(alpha_internal);
  }
  for (j in 1:H) {
    lambda_6[j] ~ cauchy(0, 1);
    phi_6[j] ~ dirichlet(alpha_internal);
  }
  //for (j in 1:H) {
  //  lambda_7[j] ~ cauchy(0, 1);
  //  phi_7[j] ~ dirichlet(alpha_internal);
  //}
  for (l in 1:(L - 1)) {
      for (j in 1:H) {
        W_internal_raw[l][, j] ~ normal(0, 1);
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
  //matrix[N, output_nodes] output_dbg = output;
  matrix[N_test, output_nodes] output_test = nn_predict(
    X_test,
    W_1,
    W_internal,
    hidden_bias,
    W_L,
    output_bias,
    L
  );
  //matrix[N_test, output_nodes] output_test_rng;
  //for (n in 1:N_test)
  //  for (j in 1:output_nodes)
  //    output_test_rng[n, j] = normal_rng(output_test[n, j], sigma);
}