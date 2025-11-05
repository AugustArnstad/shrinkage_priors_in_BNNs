// =====================
// Prior predictive model with non-centered parameterization
// =====================


data {
  int<lower=1> N;
  int<lower=1> P;
  matrix[N, P] X;
  int<lower=1> output_nodes;
  
  int<lower=1> L;
  int<lower=1> H;
  
  int<lower=1> N_test;
  matrix[N_test, P] X_test;
  
  int<lower=1> p_0;
  real<lower=0> a;
  real<lower=0> b;
  vector<lower=0>[P] alpha;
  real<lower=0, upper=1> gamma;
}

parameters {
  array[H] vector<lower=0, upper=50>[P] lambda_data;
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
  real<lower=1e-6> tau_0 = (p_0 * 1.0) / (P - p_0); //* 1 / sqrt(N);

  array[H] vector<lower=0>[P] lambda_tilde_data;
  //array[H] vector<lower=0>[P] phi_tilde_data;

  for (j in 1:H) {
    for (i in 1:P) {
      lambda_tilde_data[j][i] = fmax(1e-12, c_sq[j] * square(lambda_data[j][i]) /
                                (c_sq[j] + square(lambda_data[j][i]) * square(tau)));
    }
  }

  matrix[P, H] W_1;
  for (j in 1:H) {
    for (i in 1:P) {
      real stddev = fmax(1e-12, tau * sqrt(lambda_tilde_data[j][i]) * sqrt(phi_data[j][i]));
      W_1[i, j] = stddev * W1_raw[i, j];
    }
  }

}

model {
  tau ~ cauchy(0, tau_0);
  c_sq ~ inv_gamma(a, b);

  for (j in 1:H) {
    lambda_data[j] ~ cauchy(0, 1);
    phi_data[j] ~ dirichlet(alpha);
  }
  to_vector(W1_raw) ~ normal(0, 1);

}
