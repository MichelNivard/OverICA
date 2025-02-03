# OverICA

**Experimental!!!**

The R package `OverICA` performs overcomplete independent component analysis (ICA). Given n observations of p variables stored in matrix $y_{n,p}$, we estimate the effects of k latent variables $x_{n,k}$ on the observed variables, where k can be larger than p (overcomplete case). The basic model is:

$$y = Ax^t$$

where $A_{p,k}$ is the mixing matrix. The package also supports models with additional structure among observed variables:

$$y = (I-B)^{-1}Ax^t$$

where $B_{p,p}$ captures direct causal/structural effects among the observed variables.

## Model Overview

We opt for a distributional approach, finding a model that matches the emperical and model implied moments.

### Core Problem
Our implementation of ICA estimation requires we:
1. Matching the moments (or distributions) of model-implied data with observed data
2. Ensuring the latent variables remain independent
3. Learning the shape of non-Gaussian latent distributions

### Neural Network Solution
To avoid parametric assumptions about latent distributions, we use neural networks as flexible transformations:
1. Sample latent variables $z$ from standard normal distributions
2. Transform each $z_i$ through a neural network to create non-Gaussian latent variable $x_i$
3. Mix transformed variables using matrix $A$ to create observed variables $y$
4. Compare distributional properties between generated and observed data

### Loss Functions
We implement two approaches for distribution matching:

1. **Empirical Cumulative Generating Function (ECGF)**:
   - Captures distributional information through generalized covariance matrices
   - Computationally efficient: uses only 2nd order statistics
   - Avoids explicit computation of higher-order moments
   - Based on Podosinnikova et al. (2019)

2. **Higher-Order Moments**:
   - Explicitly computes and matches moments up to 4th order
   - Includes covariance (2nd), skewness (3rd), and kurtosis (4th)
   - More computationally intensive but potentially more precise

## Installation

```R
install.packages(c("torch", "clue", "MASS", "devtools"))
devtools::install_github("MichelNivard/OverICA")
library(OverICA)
```

## Usage

```R
# ECGF-based estimation
result <- overica(
  data = data,
  k = k,
  n_batch = 4096,
  num_t_vals = 12,      # Number of t-values for ECGF
  tbound = 0.2,         # Bounds for t-values
  lambda = 0,           # L1 regularization
  sigma = 3,            # Covariance penalty
  hidden_size = 10,
  use_adam = TRUE,
  adam_epochs = 8000,
  adam_lr = 0.1,
  use_lbfgs = FALSE,
  lbfgs_epochs = 45,
  lr_decay = 0.999
)

# Moment-based estimation
result <- overica_sem_full(
  data = data,
  k = k,
  moment_func = compute_central_moments,  # Moment computation function
  third = TRUE,                          # Include 3rd order moments
  error_cov = NULL,                      # Known error covariance
  maskB = NULL,                          # Structure constraints on B
  maskA = NULL,                          # Structure constraints on A
  lambdaA = 0.01,                        # L1 penalty on A
  lambdaB = 0.00,                        # L1 penalty on B
  sigma = 0.01                           # Covariance penalty
)
```

## Technical Details

### Central Moments
For zero-mean variables, central moments are defined as:

**Second Order (Covariance)**:
$$\text{Cov}(X_i, X_j) = E[X_iX_j]$$

**Third Order (Skewness)**:
$$E[X_iX_jX_k]$$

**Fourth Order (Kurtosis)**:
$$E[X_iX_jX_kX_l]$$

The model implied moments, and the emperical moments are used to compute the loss. The model is optimized to miimize the loss. 


### Empirical Cumulative Generating Function
For a random vector $X$, the CGF is:

$$K_X(t) = \log E[e^{t^TX}]$$

The derivatives of $K_X(t)$ evaluated at different points $t$ capture distributional information:
- First derivative: Generalized mean
- Second derivative: Generalized covariance

In `OverICA`, we:
1. Evaluate the empirical CGF at multiple points $t$
2. Match the generalized covariances between model and data
3. Use stochastic optimization to avoid overfitting to specific $t$ values

### Implementation Details
- Efficient computation using unique moment combinations
- Batch processing with torch tensors
- Optional structural constraints via mask matrices
- L1 penalties for sparse solutions
- Multiple optimization runs for stability

## References

**Podosinnikova, A.**, Perry, A., Wein, A. S., Bach, F., d'Aspremont, A., & Sontag, D. (2019). Overcomplete independent component analysis via SDP. In The 22nd international conference on artificial intelligence and statistics (pp. 2583-2592). PMLR.

**Ding, C.**, Gong, M., Zhang, K., & Tao, D. (2019). Likelihood-free overcomplete ICA and applications in causal discovery. Advances in neural information processing systems, 32.

## Related Tools

- [Likelihood Free Overcomplete ICA](https://github.com/dingchenwei/Likelihood-free_OICA)
- [https://github.com/gilgarmish/oica] Overcomplete ICA trough convex optimisation
