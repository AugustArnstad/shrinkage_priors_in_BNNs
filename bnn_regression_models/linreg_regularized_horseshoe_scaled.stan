// =====================
// Linear regression with Regularized Horseshoe Prior
// (on regression coefficients)
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
}

parameters {
  // Local scales
  vector<lower=0, upper=50>[P] lambda;

  // Global shrinkage and slab variance
  real<lower=1e-6> tau;
  real<lower=0> c_sq;

  // Non-centered regression coefficients
  vector[P] beta_raw;

  // Observation noise
  real<lower=1e-6> sigma;
}

transformed parameters {
  real<lower=1e-6> tau_0 = (1.0 * p_0 / (P - p_0)) / sqrt(N);

  // Regularized horseshoe local scales
  vector<lower=0>[P] lambda_tilde;

  // Actual regression coefficients
  vector[P] beta;

  for (i in 1:P) {
    lambda_tilde[i] = fmax(
      1e-12,
      c_sq * square(lambda[i]) /
      (c_sq + square(lambda[i]) * square(tau))
    );
    beta[i] = beta_raw[i] * fmax(1e-12, sqrt(lambda_tilde[i]) * tau);
  }
}

model {
  // Regularized horseshoe priors
  lambda ~ cauchy(0, 0.1);
  tau ~ cauchy(0, tau_0);
  c_sq ~ inv_gamma(a, b);

  // Non-centered coefficients
  beta_raw ~ normal(0, 1);

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
  // for (n in 1:N_test) {
    // y_test_rng[n] = normal_rng(y_test_mean[n], sigma);
  // }

}
