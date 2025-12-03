// =====================
// Linear regression with Regularized Horseshoe Prior
// Centered parameterization
// =====================

data {
  int<lower=1> N;                 // number of training observations
  int<lower=1> P;                 // number of covariates
  matrix[N, P] X;                 // training design matrix
  vector[N] y;                    // training response

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

  // Centered regression coefficients
  vector[P] beta;

  // Observation noise
  real<lower=1e-6> sigma;
}

transformed parameters {
  real<lower=1e-6> tau_0 = (1.0 * p_0 / (P - p_0)) / sqrt(N);

  // Regularized horseshoe local scales
  vector<lower=0>[P] lambda_tilde;

  // Per-coefficient prior std devs
  vector<lower=0>[P] beta_sd;

  for (i in 1:P) {
    lambda_tilde[i] = fmax(
      1e-12,
      c_sq * square(lambda[i]) /
      (c_sq + square(lambda[i]) * square(tau))
    );

    beta_sd[i] = fmax(
      1e-12,
      tau * sqrt(lambda_tilde[i])
    );
  }
}

model {
  // Horseshoe hyperpriors
  lambda ~ cauchy(0, 1);
  tau ~ cauchy(0, tau_0);
  c_sq ~ inv_gamma(a, b);

  // Centered prior on coefficients
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
}
