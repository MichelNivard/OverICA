devtools::document()
devtools::load_all()
library(torch)
library(MASS)
library(conclust)
# Example usage
# Generate data

n <- 50000
k <- 25
p <- 12


# 1) Generate latent Z (n x k).
z <- generate_matrix(n, k)


# 2) Define a true A (p x k)
A_true <- cbind(matrix(sample(c(0,.2,-.4,0,0,0),p*(k-p),T),p,k-p),diag(p)) # matrix(rnorm(p * k, sd=0.5), p, k)

# 3) Define a true B (p x p), ensuring the diagonal=0
B_true <- matrix(0, p, p)
# Let's put some small random off-diagonal connections
for (i in 1:p) {
  for (j in 1:p) {
    if (i != j) {
      B_true[i, j] <- sample(c(0,.3),1,prob=c(.8,.2))
    }
  }
}
# So diag(B_true) = 0. B_true is small random for demonstration.

# 4) Compute data = (I - B_true)^{-1} (Z %*% A_true)
#    We must invert (I - B_true).
I_p <- diag(p)
IB  <- I_p - B_true
# Invert:
IB_inv <- solve(IB)

# Z %*% A_true => shape (n x p) after we transpose A_true
# Actually (Z) is (n x k), A_true is (p x k), so Z %*% t(A_true) => (n x p).
ZA <- z %*% t(A_true)

# Now multiply (I - B_true)^{-1} by each row of ZA
# => data is also (n x p).
data <- t(IB_inv %*% t(ZA))

dim(data)

data <- scale(data)


# Call the estimation function
# ------------------------------------------------------------------
# Example: Using overica_sem_full on your simulated data
# ------------------------------------------------------------------

# 1) Load the torch library (and your OverICA code if not already loaded)
library(torch)
# source("overica_sem_full.R")  # <- If your code is in a separate file

# 2) Suppose we pick the "unique 4th central moments" function
#    that you already have in your package:
#    torch_unique_4th_central_moments(X)
#    We'll pass it as 'moment_func'.

# Mask the diagonal for B:

maskB <- matrix(1,p,p) - diag(p)
maskA <- cbind(matrix(1,p,k-p),diag(p))
# 3) Now run overica_sem_full()
start_time <- Sys.time()
result <- overica_sem_full(
  data        = data,     # your simulated data (n x p)
  k           = k,        # number of latent sources
  moment_func = torch_unique_4th_central_moments, 
  error_cov   = NULL,     # no additional Gaussian error
  maskB       = maskB,     # no mask => all entries of B are free
  maskA       = maskA,     # no mask => all entries of A are free
  lambdaA     = 0.005,
  lambdaB     = 0.005,
  sigma       = 0.5,
  hidden_size = 18,
  n_batch     = 8192,
  use_adam    = TRUE,
  adam_epochs = 15000,
  adam_lr     = 0.0005,
  use_lbfgs   = TRUE,
  lbfgs_epochs= 50,
  lbfgs_lr    = 1,
  lbfgs_max_iter = 20,
  lr_decay    = 1,
  clip_grad   = TRUE,
  num_runs    = 1
)

# 'result' now contains:
#  - best_result:  final B_est, A_est, etc. from the best run
#  - all_runs: all attempted runs

end_time <- Sys.time()
end_time - start_time


# How correlated are the loadings
A_est <- as.matrix(result$best_result$A_est,p,k)
A_est <- align_columns(A_est,A_true)


