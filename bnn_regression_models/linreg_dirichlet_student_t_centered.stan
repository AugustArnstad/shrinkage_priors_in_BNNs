// =====================
// Linear regression with Dirichlet–horseshoe prior
// Centered parameterization
// =====================

data {
  int<lower=1> N;                 // number of training observations
  int<lower=1> P;                 // number of covariates
  matrix[N, P] X;                 // training design matrix
  vector[N] y;                    // training response

  // int<lower=1> N_test;            // number of test observations
  // matrix[N_test, P] X_test;       // test design matrix

  int<lower=1> p_0;               // prior hyperparameters
  real<lower=0> a;
  real<lower=0> b;
  vector<lower=0>[P] alpha;       // Dirichlet concentration parameters
}

parameters {
  // Local scales (raw, before regularization)
  vector<lower=0, upper=50>[P] lambda;

  // Global scale and slab parameter
  real<lower=1e-6> tau;
  real<lower=0> c_sq;

  // Dirichlet allocation over coefficients
  simplex[P] phi_data;

  // Centered regression coefficients
  vector[P] beta;

  // Observation noise
  real<lower=1e-6> sigma;
}

transformed parameters {
  real<lower=1e-6> tau_0 = (p_0 * 1.0) / (P - p_0) * 1 / sqrt(N);

  // Regularized local scales
  vector<lower=0>[P] lambda_tilde;

  // Per-coefficient prior std devs (scales)
  vector<lower=0>[P] beta_sd;

  for (i in 1:P) {
    lambda_tilde[i] = fmax(
      1e-12,
      c_sq * square(lambda[i]) /
      (c_sq + square(lambda[i]) * square(tau))
    );

    beta_sd[i] = fmax(
      1e-12,
      tau * sqrt(lambda_tilde[i]) * sqrt(phi_data[i])
    );
  }
}

model {
  // Hyperpriors
  tau ~ cauchy(0, tau_0);
  c_sq ~ inv_gamma(a, b);

  // Local scales: heavy-tailed t(3)
  lambda ~ student_t(3, 0, 1);

  // Dirichlet allocation over coefficients
  phi_data ~ dirichlet(alpha);

  // Centered prior for coefficients
  for (i in 1:P)
    beta[i] ~ normal(0, beta_sd[i]);

  // Noise prior
  sigma ~ inv_gamma(3, 2);

  // Likelihood
  y ~ normal(X * beta, sigma);
}

generated quantities {
  // Fitted values on training data
  vector[N] y_hat = X * beta;

  // Predictive mean and draws on test data
  // vector[N_test] y_test_mean;
  // vector[N_test] y_test_rng;

  // y_test_mean = X_test * beta;
  // for (n in 1:N_test)
  //   y_test_rng[n] = normal_rng(y_test_mean[n], sigma);
}
