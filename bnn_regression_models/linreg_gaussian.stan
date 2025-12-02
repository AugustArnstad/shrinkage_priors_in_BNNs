// =====================
// Linear regression with Gaussian prior
// =====================

data {
  int<lower=1> N;                 // number of training observations
  int<lower=1> P;                 // number of covariates
  matrix[N, P] X;                 // training design matrix
  vector[N] y;                    // training response

  // int<lower=1> N_test;            // number of test observations
  // matrix[N_test, P] X_test;       // test design matrix
}

parameters {
  vector[P] beta;                 // regression coefficients
  real<lower=1e-6> sigma;         // observation noise
}

transformed parameters {
  // Linear predictor for training data
  vector[N] f = X * beta;
}

model {
  // Gaussian prior on coefficients, scaled by 1/sqrt(P)
  beta ~ normal(0, inv_sqrt(P));

  // Prior on noise
  sigma ~ inv_gamma(3, 2);

  // Likelihood
  y ~ normal(f, sigma);
}

generated quantities {
  // Fitted mean on training data
  vector[N] y_hat = f;

  // Predictive mean and draws on test data
  // vector[N_test] y_test_mean;
  // vector[N_test] y_test_rng;

  // y_test_mean = X_test * beta;
  // for (n in 1:N_test)
    // y_test_rng[n] = normal_rng(y_test_mean[n], sigma);
}
