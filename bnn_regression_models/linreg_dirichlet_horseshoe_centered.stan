// ============================================
// Linear regression with Dirichlet–horseshoe / DSM prior on coefficients
// Centered parameterization for beta
// ============================================

data {
  int<lower=1> N;                // number of observations
  int<lower=1> P;                // number of covariates
  matrix[N, P] X;                // design matrix
  vector[N] y;                   // response

  // Prior hyperparameters (same structure as in the NN model)
  int<lower=1> p_0;
  real<lower=0> a;
  real<lower=0> b;
  vector<lower=0>[P] alpha;      // Dirichlet concentration for phi
}

parameters {
  // Local scales (raw, before regularization)
  vector<lower=0, upper=50>[P] lambda_data;

  // Dirichlet weights over coefficients (one simplex for all P coeffs)
  simplex[P] phi_data;

  // Global scale and slab parameter
  real<lower=1e-6> tau;
  real<lower=1e-6> c_sq;

  // Centered regression coefficients
  vector[P] beta;

  // Noise scale
  real<lower=1e-6> sigma;
}

transformed parameters {
  // Same tau_0 as in your NN model
  real<lower=1e-6> tau_0 = (p_0 * 1.0) / (P - p_0) * 1 / sqrt(N);

  // Regularized local scales
  vector<lower=0>[P] lambda_tilde_data;

  // Per-coefficient prior std devs (scale parameters)
  vector<lower=0>[P] beta_sd;

  for (i in 1:P) {
    lambda_tilde_data[i] = fmax(
      1e-12,
      c_sq * square(lambda_data[i]) /
      (c_sq + square(lambda_data[i]) * square(tau))
    );

    beta_sd[i] = fmax(
      1e-12,
      tau * sqrt(lambda_tilde_data[i]) * sqrt(phi_data[i])
    );
  }
}

model {
  // Hyperpriors
  tau ~ cauchy(0, tau_0);
  c_sq ~ inv_gamma(a, b);

  lambda_data ~ cauchy(0, 1);
  phi_data ~ dirichlet(alpha);

  // Centered prior on coefficients
  for (i in 1:P)
    beta[i] ~ normal(0, beta_sd[i]);

  sigma ~ inv_gamma(3, 2);

  // Likelihood (vectorized)
  y ~ normal(X * beta, sigma);
}

generated quantities {
  vector[N] y_hat;

  // Fitted mean
  y_hat = X * beta;
}
