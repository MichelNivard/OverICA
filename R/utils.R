
#' Compute Unique Indices for 4th, 3rd, and 2nd Central Moments
#'
#' This function precomputes the unique index combinations needed to calculate the
#' central moments. It generates:
#' - 4th-order moment indices: (i, j, k, l) where i <= j <= k <= l
#' - 3rd-order moment indices: (i, j, k) where i <= j <= k
#' - 2nd-order moment indices: (i, j) where i <= j
#'
#' Precomputing these indices allows for efficient moment calculation in optimization loops.
#'
#' @param p An integer representing the number of features (columns in X).
#' @return A list containing three tensors:
#'   - `indices_2nd`: A (num_2nd, 2) tensor with unique 2nd-order moment indices.
#'   - `indices_3rd`: A (num_3rd, 3) tensor with unique 3rd-order moment indices.
#'   - `indices_4th`: A (num_4th, 4) tensor with unique 4th-order moment indices.
#' @export
compute_unique_moment_indices <- function(p) {
  # 2nd-order indices (i <= j)
  indices_2nd <- list()
  for (i in seq_len(p)) {
    for (j in i:p) {
      indices_2nd[[length(indices_2nd) + 1]] <- c(i, j)
    }
  }
  indices_2nd <- torch_tensor(do.call(rbind, indices_2nd))
  # 3rd-order indices (i <= j <= k)
  indices_3rd <- list()
  for (i in seq_len(p)) {
    for (j in i:p) {
      for (k in j:p) {
        indices_3rd[[length(indices_3rd) + 1]] <- c(i, j, k)
      }
    }
  }
  indices_3rd <- torch_tensor(do.call(rbind, indices_3rd)) 
  
  # 4th-order indices (i <= j <= k <= l)
  indices_4th <- list()
  for (i in seq_len(p)) {
    for (j in i:p) {
      for (k in j:p) {
        for (l in k:p) {
          indices_4th[[length(indices_4th) + 1]] <- c(i, j, k, l)
        }
      }
    }
  }
  indices_4th <- torch_tensor(do.call(rbind, indices_4th)) 
  
  return(list(
    indices_2nd = indices_2nd,
    indices_3rd = indices_3rd,
    indices_4th = indices_4th
  ))
}


#' Compute Unique 4th, 3rd, and 2nd Central Moments Using Precomputed Indices
#'
#' Given a dataset X of shape (n, p), this function computes the unique central moments:
#' - 4th-order moments: E[(X_i - mu_i)(X_j - mu_j)(X_k - mu_k)(X_l - mu_l)] for i <= j <= k <= l
#' - 3rd-order moments: E[(X_i - mu_i)(X_j - mu_j)(X_k - mu_k)] for i <= j <= k
#' - 2nd-order moments: E[(X_i - mu_i)(X_j - mu_j)] for i <= j
#'
#' Instead of computing all possible moment combinations, this function uses precomputed indices
#' (from `compute_unique_moment_indices()`) to extract only the required values, significantly
#' improving computational efficiency.
#'
#' The results are concatenated into a single 1D torch tensor in the order of 4th, 3rd, and 2nd-order moments.
#'
#' @param X A torch tensor of shape (n, p).
#' @param indices A list of precomputed index tensors from `compute_unique_moment_indices()`, containing:
#'   - `indices_2nd`: Indices for unique 2nd-order moments.
#'   - `indices_3rd`: Indices for unique 3rd-order moments.
#'   - `indices_4th`: Indices for unique 4th-order moments.
#' @param third A logical indicating wherther thris order moments are to be considered
#' @return A 1D torch tensor containing all unique moments, concatenated in the order of:
#'   - 4th-order moments
#'   - 3rd-order moments
#'   - 2nd-order moments
#' @export
compute_central_moments <- function(X, indices,third=TRUE) {
  n <- X$size(1)
  p <- X$size(2)
  
  # Center the data
  means <- X$mean(dim = 1)
  Xc <- X - means$view(c(1, p))
  
  # Compute 2nd-order moments
  i2 <- indices$indices_2nd[, 1]
  j2 <- indices$indices_2nd[, 2]
  unique_2nd <- (Xc[, i2] * Xc[, j2])$mean(dim = 1)
  
  if(third == TRUE){
  # Compute 3rd-order moments
  i3 <- indices$indices_3rd[, 1]
  j3 <- indices$indices_3rd[, 2]
  k3 <- indices$indices_3rd[, 3]
  unique_3rd <- (Xc[, i3] * Xc[, j3] * Xc[, k3])$mean(dim = 1)
  }
  # Compute 4th-order moments
  i4 <- indices$indices_4th[, 1]
  j4 <- indices$indices_4th[, 2]
  k4 <- indices$indices_4th[, 3]
  l4 <- indices$indices_4th[, 4]
  unique_4th <- (Xc[, i4] * Xc[, j4] * Xc[, k4] * Xc[, l4])$mean(dim = 1)
  
  if(third==TRUE){
  # Concatenate all moments into a single vector
  torch_cat(c(unique_2nd, unique_3rd, unique_4th))
  }else{
  torch_cat(c(unique_2nd, unique_4th)) 
  }
  }

#' Convert Moments Vector to Covariance Matrix and Extract Kurtosis
#'
#' This function takes the output of `compute_central_moments()` and reconstructs:
#' - The **covariance matrix** from the 2nd-order moments (only the lower triangular part).
#' - The **kurtosis** values extracted from the co-kurtosis vector (4th-order moments).
#'
#' If 3rd-order moments were included in the original computation, this function
#' correctly accounts for their presence when extracting 4th-order moments.
#'
#' @param moments A 1D torch tensor of concatenated moments from `compute_central_moments()`.
#' @param indices A list of precomputed index tensors from `compute_unique_moment_indices()`, containing:
#'   - `indices_2nd`: Indices for unique 2nd-order moments.
#'   - `indices_4th`: Indices for unique 4th-order moments.
#' @param p Number of observed variables
#' @param third A logical indicating whether 3rd-order moments were included.
#' @return A list containing:
#'   - `cov_matrix`: A (p, p) lower triangular covariance matrix reconstructed from the 2nd-order moments.
#'   - `kurtosis`: A (p,) tensor of kurtosis values extracted from the 4th-order moments.
#' @export
extract_covariance_and_kurtosis <- function(moments, indices,p, third=TRUE) {
 
  
  # Determine where 4th-order moments start in the vector
  num_2nd <- indices$indices_2nd$size(1)
  num_3rd <- if (third) indices$indices_3rd$size(1) else 0
  start_4th <- num_2nd + num_3rd  # Start index for 4th-order moments

  # Extract 2nd-order moments (covariance values)
  cov_matrix <- torch_zeros(p, p)
  for (idx in seq_len(num_2nd)) {
    i <- indices$indices_2nd[idx, 1]$item()
    j <- indices$indices_2nd[idx, 2]$item() 
    cov_matrix[i, j] <- moments[idx]
    cov_matrix[j, i] <- moments[idx]  # Ensure symmetry
  }
  
  # Extract 4th-order moments (co-kurtosis)
  kurtosis <- torch_zeros(p)
  for (idx in seq_len(indices$indices_4th$size(1))) {
    i <- indices$indices_4th[idx, 1]$item() 
    j <- indices$indices_4th[idx, 2]$item() 
    k <- indices$indices_4th[idx, 3]$item() 
    l <- indices$indices_4th[idx, 4]$item() 
    if (i == j && j == k && k == l) {  # Extract diagonal elements (self-kurtosis)
      kurtosis[i] <- moments[start_4th + idx]
    }
  }

  return(list(
    cov_matrix = cov_matrix,
    kurtosis = kurtosis
  ))
}

#' Align Estimated and True Matrices Using the Hungarian Algorithm
#'
#' This function aligns the columns of an estimated matrix with those of the true matrix by maximizing the absolute correlation between columns using the Hungarian algorithm.
#'
#' @param D_est A numeric matrix of estimated components (n x k).
#' @param D_true A numeric matrix of true components (n x k).
#' @return A numeric matrix with columns of \code{D_est} aligned to \code{D_true}.
#' @importFrom clue solve_LSAP
#' @export
#' @examples
#' # Assuming D_est and D_true are matrices
#' D_aligned <- align_columns(D_est, D_true)
align_columns <- function(D_est, D_true) {
  k <- ncol(D_true)
  cost_matrix <- matrix(0, nrow = k, ncol = k)
  for (i in 1:k) {
    for (j in 1:k) {
      cost_matrix[i, j] <- abs(cor(D_true[, i], D_est[, j]))
    }
  }
  assignment <- solve_LSAP(cost_matrix, maximum = TRUE)
  D_est_aligned <- D_est[, assignment]

  correlations <- diag(cor(D_est_aligned, D_true))
  print(correlations)
  print(sum(abs(correlations) > .9))

  for (i in 1:ncol(D_est_aligned)) {
    D_est_aligned[, i] <- sign(correlations[i]) * D_est_aligned[, i]
  }

  return(D_est_aligned)
}

#' Compute Empirical Cumulative Generating Function (ECGF) and its Derivatives
#'
#' This function calculates the Empirical Cumulative Generating Function (ECGF) and its first
#' and second derivatives using a batch of data. It is designed to work with torch tensors.
#'
#' @param t_vectors A tensor of shape \code{(num_t_vals, p)}, where \code{num_t_vals} is the number of t-values and \code{p} is the dimensionality.
#' @param data_tensor A tensor of shape \code{(n, p)}, where \code{n} is the number of data points and \code{p} is the dimensionality.
#'
#' @return A list containing:
#' \item{K}{A tensor of shape \code{(num_t_vals)} representing the ECGF values.}
#' \item{dK}{A tensor of shape \code{(num_t_vals, p)} representing the first derivative of the ECGF.}
#' \item{d2K}{A tensor of shape \code{(num_t_vals, p, p)} representing the second derivative of the ECGF.}
#'
#' @details
#' The function computes the ECGF using the dot product between the data tensor and t-vectors, followed by the calculation of log-sum-exp for numerical stability.
#' The first derivative \code{dK} is obtained using weighted averages of the data points. The second derivative \code{d2K} is computed through a batch of weighted centered data products.
#'
#' @examples
#' \dontrun{
#' # Assume torch and appropriate tensors are loaded
#' t_vectors <- torch_randn(c(5, 3)) # 5 t-values, 3-dimensional space
#' data_tensor <- torch_randn(c(100, 3)) # 100 data points, 3-dimensional space
#' result <- torch_ecgf_batch(t_vectors, data_tensor)
#' print(result$K)  # ECGF values
#' print(result$dK) # First derivative
#' print(result$d2K) # Second derivative
#' }
#'
#' @export
torch_ecgf_batch <- function(t_vectors, data_tensor) {
  # t_vectors: (num_t_vals x p)
  # data_tensor: (n x p)

  num_t_vals <- t_vectors$size(1)
  n <- data_tensor$size(1)
  p <- data_tensor$size(2)

  # Compute the dot product between data and t_vectors
  dot_product <- data_tensor$matmul(t_vectors$transpose(1, 2))  # (n x num_t_vals)

  # Compute K_t
  K_t <- torch_logsumexp(dot_product, dim = 1) - log(n)  # (num_t_vals)

  # Compute log weights for numerical stability
  log_weights <- dot_product - torch_logsumexp(dot_product, dim = 1, keepdim = TRUE)  # (n x num_t_vals)
  weights <- torch_exp(log_weights)  # (n x num_t_vals)

  # Compute first derivative dK
  # Expand weights to (n x num_t_vals x 1)
  weights_expanded <- weights$unsqueeze(3)  # (n x num_t_vals x 1)

  # Expand data_tensor to (n x 1 x p)
  data_expanded <- data_tensor$unsqueeze(2)  # (n x 1 x p)

  # Multiply and sum over n
  data_weighted <- weights_expanded * data_expanded  # (n x num_t_vals x p)

  # Sum over n
  dK <- torch_sum(data_weighted, dim = 1)  # (num_t_vals x p)

  # Compute second derivative d2K
  # Expand dK to (1 x num_t_vals x p)
  dK_expanded <- dK$unsqueeze(1)  # (1 x num_t_vals x p)

  data_centered <- data_expanded - dK_expanded  # (n x num_t_vals x p)

  # Compute weighted_data_centered
  weighted_data_centered <- weights_expanded * data_centered  # (n x num_t_vals x p)

  # Compute d2K using batched matrix multiplication
  # Permute dimensions to (num_t_vals x n x p)
  data_centered_per_t <- data_centered$permute(c(2, 1, 3))  # (num_t_vals x n x p)
  weighted_data_centered_per_t <- weighted_data_centered$permute(c(2, 1, 3))  # (num_t_vals x n x p)

  # Transpose weighted_data_centered_per_t to (num_t_vals x p x n)
  weighted_centered_transposed <- weighted_data_centered_per_t$transpose(2, 3)  # (num_t_vals x p x n)

  # Compute d2K_batch using torch_bmm
  d2K_batch <- torch_bmm(weighted_centered_transposed, data_centered_per_t)  # (num_t_vals x p x p)

  return(list(K = K_t, dK = dK, d2K = d2K_batch))
}

#' Average Multiple OICA Runs to Obtain a Consensus Mixing Matrix
#'
#' This function averages the estimated mixing matrices from multiple runs of the `overica` function.
#' It aligns components across runs, accounting for sign ambiguity inherent in ICA, by clustering the
#' components and combining sign-flipped versions to obtain a consensus mixing matrix.
#'
#' @param result A list object returned by the `overica` function with multiple runs, containing the estimated mixing matrices.
#' @param num_runs An integer specifying the number of runs.
#' @param p An integer specifying the number of observed variables.
#' @param k An integer specifying the number of latent variables.
#' @param maxit Maximum number of iterations for the constrained clustering algorithm
#' @return A matrix representing the averaged mixing matrix after aligning and averaging over runs.
#' @importFrom stats density cor
#' @importFrom dplyr distinct
#' @examples
#' \dontrun{
#' # Assuming 'result' is the output from overica with multiple runs
#' p <- ncol(data)
#' k <- 5
#' num_runs <- 10
#' averaged_A <- avgOICAruns(result, num_runs, p, k)
#' }
#' @export
avgOICAruns <- function(result, num_runs, p, k,maxit=2000) {

  # Retrieve the estimated A matrix from the best result
  A_est <- as.matrix(result$best_result$A_est, p, k)

  # Initialize A_base with the A matrix from the first run
  A_base <- as.matrix(result$all_runs[[1]]$A_est, p, k)
  A_base <- cbind(align_columns(as.matrix(result$all_runs[[2]]$A_est, p, k),A_base),A_base)
  # Loop through the remaining runs and append each A matrix to A_base
  for (i in 3:num_runs) {
    A_new <- result$all_runs[[i]]$A_est
    A_base <- cbind(as.matrix(A_new, p, k), A_base)
  }

  # Combine A_base with its negated version to allow for sign-flip ambiguity
  A_base <- cbind(A_base, -1 * A_base)

  # Transpose A_base for further analysis
  A_base_t <- t(A_base)

  # Function to compute the mode of a continuous variable using Kernel Density Estimation (KDE)
  mode_kde <- function(x, bw = "nrd0") {
    # Ensure x is numeric and remove NA values
    x <- na.omit(as.numeric(x))

    # Estimate the density of x using the specified bandwidth method
    dens <- density(x, bw = bw)

    # Find the mode by locating the point with the highest density
    mode_value <- dens$x[which.max(dens$y)]

    return(mode_value)
  }

  # Set dimensions for clustering and merging process
  k2 <- 2 * k
  nr2 <- 2 * num_runs

  # Create matrix of pairwise indices for the clustering process
  a <- cbind(rep(1:k, k2), rep(1:k, each = k2))
  for (i in 2:nr2) {
    a <- rbind(a, cbind(rep(((i - 1) * k + 1):(i * k), k2), rep(((i - 1) * k + 1):(i * k), each = k2)))
  }

  # Perform clustering using the ckmeans algorithm with mustLink and cantLink constraints
  clust <- ckmeans(A_base_t, mustLink = matrix(c(1, k + 1), nrow = 1), cantLink = a, k = k2, maxIter = maxit)

  # Initialize a matrix to store the median of the clustered A matrix
  A_med <- matrix(NA, p, k2)

  # Compute the mode for each cluster and store it in A_med
  for (i in 1:k2) {
    A_med[,i] <- apply(A_base_t[clust == i,], 2, median)
  }

  # Calculate the correlation matrix of A_med to identify loadings with high correlations
  cor <- cor(A_med)
  pairs <- matrix(NA, k2, 2)

  # Find pairs of loadings with the smallest correlation and store their indices
  for (i in 1:k2) {
    w <- which(cor[,i] == min(cor[,i]))
    pairs[i,] <- c(w, i)
  }

  # Generate a unique index for each pair and remove duplicates
  index <- pairs[,1] * pairs[,2]
  pairs <- cbind.data.frame(pairs, index)
  pairs <- distinct(pairs, index, .keep_all = TRUE)

  # Remove pairs where the two indices are equal (no need to merge identical components)
  pairs <- pairs[pairs[,1] != pairs[,2],]

  # Initialize matrix to store the merged median A matrix
  A_med_merge <- matrix(NA, nrow = p, ncol = k)

  # Merge the most correlated pairs by averaging them, adjusting for sign flips
  for (i in 1:k) {
    A_med_merge[,i] <- (A_med[,pairs[i,1]] + -1 * A_med[,pairs[i,2]]) / 2
  }

  # Return the final merged matrix of median loadings
  out <- list(A_med = A_med_merge,A_all = A_base_t)
  return(out)
}


#' Compute pseudo AIC for an OLS-style reconstruction loss
#'
#' Given the mean-squared reconstruction error (raw_loss), the estimated matrices A and B,
#' and the number of observations (n_used), this function computes the sum of squared errors (SSE)
#' as SSE = raw_loss * n_used and counts the effective number of parameters (k_eff) as the number of
#' entries in A and B whose absolute value exceeds a given threshold.
#' Then the AIC is computed as:
#'
#'    AIC = n_used * log(SSE/n_used) + 2 * k_eff
#'
#' @param raw_loss A numeric value representing the mean squared error (MSE) of the reconstruction.
#' @param A_est A torch tensor (or array) for the estimated matrix A.
#' @param B_est A torch tensor (or array) for the estimated matrix B.
#' @param n_used Number of observations used in the loss calculation.
#' @param threshold Numeric; entries smaller than this in absolute value are treated as zero.
#' @param third Are third moments to be considered?
#'
#' @return A list with elements:
#'   \item{AIC}{The computed AIC value.}
#'   \item{SSE}{The sum of squared errors.}
#'   \item{k_eff}{The effective number of parameters.}
#'
#' @export
compute_AIC <- function(raw_loss, A_est, B_est,k, threshold = 0.001,third=TRUE) {
  # raw_loss is assumed to be the MSE (i.e. SSE/moments) 
  SSE <- raw_loss * ((k * (k-1))/2 + (k * (k-1) * (k-2)) /6  + (k * (k-1) * (k-2) * (k-3)) / 24)

  if(third==TRUE){
    n_moments <- ((k * (k-1))/2 + (k * (k-1) * (k-2)) /6  + (k * (k-1) * (k-2) * (k-3)) / 24)
  } else{
    n_moments <- ((k * (k-1))/2 + (k * (k-1) * (k-2) * (k-3)) / 24)

  }
  # Convert A_est and B_est to R arrays if they are torch tensors.
  if (inherits(A_est, "torch_tensor")) A_est <- as.array(A_est)
  if (inherits(B_est, "torch_tensor")) B_est <- as.array(B_est)
  
  # Count effective parameters: those with absolute value greater than threshold.
  k_eff <- sum(abs(A_est) > threshold) + sum(abs(B_est) > threshold) 
  AIC_val <- n_moments * log(SSE / n_moments) + (2 * k_eff)
  
  return(list(AIC = AIC_val, SSE = SSE, k_eff = k_eff,n_moments=n_moments))
}


#' Compute pseudo BIC for an OLS-style reconstruction loss
#'
#' Given the mean-squared reconstruction error (raw_loss), the estimated matrices A and B,
#' and the number of observations (n_used), this function computes the sum of squared errors (SSE)
#' as SSE = raw_loss * n_used and counts the effective number of parameters (k_eff) as the number of
#' entries in A and B whose absolute value exceeds a given threshold.
#' Then the BIC is computed as:
#'
#'    BIC = n_used * log(SSE/n_used) + log(n_used) * k_eff
#'
#' @param raw_loss A numeric value representing the mean squared error (MSE) of the reconstruction.
#' @param A_est A torch tensor (or array) for the estimated matrix A.
#' @param B_est A torch tensor (or array) for the estimated matrix B.
#' @param n_used Number of observations used in the loss calculation.
#' @param threshold Numeric; entries smaller than this in absolute value are treated as zero.
#' @param third Are third moments to be considered?
#'
#' @return A list with elements:
#'   \item{BIC}{The computed BIC value.}
#'   \item{SSE}{The sum of squared errors.}
#'   \item{k_eff}{The effective number of parameters.}
#'
#' @export
compute_BIC <- function(raw_loss, A_est, B_est,k, threshold = 0.001,third=TRUE) {
  # raw_loss is assumed to be the MSE (i.e. SSE/moments) 
  SSE <- raw_loss * ((k * (k-1))/2 + (k * (k-1) * (k-2)) /6  + (k * (k-1) * (k-2) * (k-3)) / 24)
  if(third==TRUE){
    n_moments <- ((k * (k-1))/2 + (k * (k-1) * (k-2)) /6  + (k * (k-1) * (k-2) * (k-3)) / 24)
  } else{
    n_moments <- ((k * (k-1))/2 + (k * (k-1) * (k-2) * (k-3)) / 24)

  }
  # Convert A_est and B_est to R arrays if they are torch tensors.
  if (inherits(A_est, "torch_tensor")) A_est <- as.array(A_est)
  if (inherits(B_est, "torch_tensor")) B_est <- as.array(B_est)
  
  # Count effective parameters: those with absolute value greater than threshold.
  k_eff <- sum(abs(A_est) > threshold) + sum(abs(B_est) > threshold) 
  BIC_val <- n_moments  * log(SSE / n_moments)  + (log(n_moments) * k_eff)
  
  return(list(BIC = BIC_val, SSE = SSE, k_eff = k_eff,n_moments=n_moments))
}
