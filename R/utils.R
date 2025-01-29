
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
      kurtosis[i] <- moments[start_4th + idx - 1]
    }
  }

  return(list(
    cov_matrix = cov_matrix,
    kurtosis = kurtosis
  ))
}

