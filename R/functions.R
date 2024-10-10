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



#' Estimate an Overcomplete Independent Component Analysis (OICA) Using Torch and Empirical Cumulant Generating Functions
#'
#' This function estimates parameters of an OICA model using torch tensors and ECGFs. It optimizes the parameters using a combination of Adam and L-BFGS optimizers.
#'
#' @param data A numeric matrix representing the observed data (n x p).
#' @param k An integer specifying the number of latent variables.
#' @param num_t_vals An integer specifying the number of t-values to use (default is 8).
#' @param lambda A numeric value for the L1 regularization parameter (default is 0.01).
#' @param sigma A numeric value for the covariance penalty parameter (default is 0.01).
#' @param tbound A numeric value specifying the standard deviation for t-values (default is 0.2).
#' @param abound A numeric value that specifies the upper and lower bound of the uniformly sampled starting values for A (default is 0.5).
#' @param n_batch An integer specifying the batch size for the neural network (default is 1024).
#' @param hidden_size An integer specifying the hidden size of the neural network (default is 10).
#' @param neural_nets User specified or previously fitted neural nets; if left at NULL, the model will initialize new untrained neural networks.
#' @param clip_grad Should the gradient be clipped during optimization?
#' @param use_adam Logical; whether to use the Adam optimizer first (default is TRUE).
#' @param adam_epochs An integer specifying the number of epochs for Adam optimizer (default is 100).
#' @param adam_lr A numeric value specifying the learning rate for Adam optimizer (default is 0.01).
#' @param use_lbfgs Logical; whether to use the L-BFGS optimizer after Adam (default is TRUE).
#' @param lbfgs_epochs An integer specifying the number of epochs for L-BFGS optimizer (default is 50).
#' @param lbfgs_lr A numeric value specifying the learning rate for L-BFGS optimizer (default is 1).
#' @param lbfgs_max_iter An integer specifying the max iterations per optimization step for L-BFGS (default is 20).
#' @param lr_decay A numeric value specifying the decay rate for the multiplicative learning rate decay; set to 1 for no decay (default is 0.999).
#' @param num_runs An integer specifying the number of times to run the estimation (default is 1).
#' @return A list containing the best result (\code{best_result}) and a list of all runs (\code{all_runs}).
#' @import torch
#' @export
#' @examples
#' # Assuming 'data' is a matrix of observed data
#' result <- overica(data, k = 5, num_runs = 10)
overica <- function(
    data,
    k,
    num_t_vals = 8,
    lambda = 0.01,
    sigma = 0.01,
    tbound = 0.2,
    abound = 0.5,
    n_batch = 1024,
    hidden_size = 10,
    neural_nets = NULL,
    clip_grad = TRUE,
    use_adam = TRUE,       # Whether to use Adam optimizer first
    adam_epochs = 100,     # Number of epochs for Adam optimizer
    adam_lr = 0.01,        # Learning rate for Adam optimizer
    use_lbfgs = TRUE,      # Whether to use L-BFGS optimizer after Adam
    lbfgs_epochs = 50,     # Number of epochs for L-BFGS optimizer
    lbfgs_lr = 1,          # Learning rate for L-BFGS optimizer
    lbfgs_max_iter = 20,   # Max iterations per optimization step for L-BFGS
    lr_decay = 0.999,      # Decay rate for multiplicative learning rate decay, set to 1 for no decay
    num_runs = 1           # Number of times to run the estimation
) {
  # Check if CUDA is available
  device <- if (cuda_is_available()) torch_device("cuda") else torch_device("cpu")

  # Data tensor
  data_tensor <- torch_tensor(data, dtype = torch_float(), device = device)

  n <- nrow(data)
  p <- ncol(data)

  # Sample t_vals_o once outside fn()
  t_vals_o <- matrix(0, nrow = 1, ncol = p)
  additional_t_vals <- matrix(rnorm((num_t_vals - 1) * p, 0, tbound), ncol = p)
  t_vals_o <- rbind(t_vals_o, additional_t_vals)
  t_vals_o_tensor <- torch_tensor(t_vals_o, dtype = torch_float(), device = device)

  # Compute data_dK and data_d2K once
  ecgf_result_data <- torch_ecgf_batch(t_vals_o_tensor, data_tensor)
  data_dK <- ecgf_result_data$dK  # (num_t_vals x p)
  data_d2K <- ecgf_result_data$d2K  # (num_t_vals x p x p)

  # Covariance penalty function
  covariance_penalty <- function(s) {
    n <- s$size(1)
    cov_matrix <- (s$t()$matmul(s)) / (n - 1)
    cov_penalty <- torch_sum(torch_square(cov_matrix - torch_eye(s$size(2), device = device)))
    return(cov_penalty)
  }

  # Covariance matrix function
  covariance_matrix <- function(s) {
    n <- s$size(1)
    cov_matrix <- (s$t()$matmul(s)) / (n - 1)
    return(cov_matrix)
  }

  # Initialize lists to store all runs
  all_runs <- vector('list', num_runs)
  best_loss <- Inf
  best_result <- NULL

  for (run_idx in 1:num_runs) {
    cat("Run", run_idx, "/", num_runs, "\n")

    # Initialize parameters
    A_params <- torch_tensor(runif(p * k, min = -abound, max = abound), requires_grad = TRUE, device = device)

    NeuralNets <- nn_module(
      initialize = function(k, hidden_size) {
        self$nets <- nn_module_list()
        for (i in 1:k) {
          self$nets$append(
            nn_sequential(
              nn_linear(1, hidden_size),
              nn_relu(),
              nn_linear(hidden_size, 1)
            )
          )
        }
      },
      forward = function(z) {
        # z is of shape (n_batch, k)
        s_list <- list()
        for (i in 1:k) {
          z_i <- z[, i, drop = FALSE]
          s_list[[i]] <- self$nets[[i]](z_i)
        }
        s <- torch_cat(s_list, dim = 2)
        return(s)
      }
    )

    # Initialize neural_nets
    if (is.null(neural_nets)) {
      neural_nets_run <- NeuralNets$new(k = k, hidden_size = hidden_size)$to(device = device)
    } else {
      neural_nets_run <- neural_nets$to(device = device)
    }

    # Parameters to optimize
    params <- c(list(A_params), neural_nets_run$parameters)

    # Sample z ~ N(0,I) once and keep it fixed
    z_fixed <- torch_randn(c(n_batch, k), device = device)

    # Define the objective function
    fn <- function() {
      # Reshape A_params into a p x k matrix
      A <- A_params$view(c(p, k))

      # Use fixed z
      z <- z_fixed

      # Pass z through neural_nets to get s
      s <- neural_nets_run(z)

      # Compute data_hat = s %*% t(A)
      data_hat <- s$matmul(A$t())

      # Compute model's K(t), dK(t), d2K(t) using data_hat
      ecgf_result_model <- torch_ecgf_batch(t_vals_o_tensor, data_hat)
      model_dK <- ecgf_result_model$dK  # (num_t_vals x p)
      model_d2K <- ecgf_result_model$d2K  # (num_t_vals x p x p)

      # Compute the differences and accumulate the losses
      diff_dK <- model_dK - data_dK  # (num_t_vals x p)
      loss_dK <- torch_sum(diff_dK * diff_dK)

      diff_d2K <- model_d2K - data_d2K  # (num_t_vals x p x p)
      loss_d2K <- torch_sum(diff_d2K * diff_d2K)

      # Total loss
      loss <- loss_dK + loss_d2K

      # Add regularization terms
      loss <- loss + lambda * torch_sum(torch_abs(A)) + sigma * covariance_penalty(s)

      return(loss)
    }

    # Initialize lists to store losses
    adam_losses <- numeric(0)
    lbfgs_losses <- numeric(0)

    # Adam optimizer
    if (use_adam) {
      optimizer_adam <- optim_adam(params = params, lr = adam_lr)
      scheduler <- lr_multiplicative(optimizer_adam, lr_lambda = function(epoch) lr_decay)
      adam_losses <- numeric(adam_epochs)

      for (epoch in 1:adam_epochs) {
        optimizer_adam$zero_grad()
        loss_value <- fn()
        loss_value$backward()
        if (clip_grad) {
          nn_utils_clip_grad_norm_(params, max_norm = 2)
        }
        optimizer_adam$step()
        scheduler$step()
        adam_losses[epoch] <- as.numeric(loss_value$item())

        # Optional: Print progress
        if (epoch %% 10 == 0 || epoch == adam_epochs) {
          current_lr <- optimizer_adam$param_groups[[1]]$lr
          cat("\r", "Run:", run_idx, "Adam Epoch:", epoch, "Loss:", adam_losses[epoch], "LR:", current_lr)
        }
      }
      cat("\n")
    }

    # L-BFGS optimizer
    if (use_lbfgs) {
      optimizer_lbfgs <- optim_lbfgs(
        params = params,
        lr = lbfgs_lr,
        max_iter = lbfgs_max_iter,
        history_size = 10,
        line_search_fn = "strong_wolfe"
      )
      scheduler_lbfgs <- lr_multiplicative(optimizer_lbfgs, lr_lambda = function(epoch) lr_decay)
      lbfgs_losses <- numeric(lbfgs_epochs)

      for (epoch in 1:lbfgs_epochs) {
        closure <- function() {
          optimizer_lbfgs$zero_grad()
          loss <- fn()
          loss$backward()
          return(loss)
        }
        loss_value <- optimizer_lbfgs$step(closure)
        scheduler_lbfgs$step()
        lbfgs_losses[epoch] <- as.numeric(loss_value$item())

        # Optional: Print progress
        if (epoch %% 1 == 0) {
          current_lr <- optimizer_lbfgs$param_groups[[1]]$lr
          cat("\r", "Run:", run_idx, "L-BFGS Epoch:", epoch, "Loss:", lbfgs_losses[epoch], "LR:", current_lr)
        }
      }
      cat("\n")
    }

    # Compute the final loss value
    final_loss <- as.numeric(fn()$item())

    # Retrieve optimized parameters
    A_est <- A_params$view(c(p, k))
    cov_est <- covariance_matrix(neural_nets_run(z_fixed))

    # Save the current result
    current_result <- list(
      A_est = A_est,
      cov = cov_est,
      neural_nets = neural_nets_run,
      adam_losses = if (use_adam) adam_losses else NULL,
      lbfgs_losses = if (use_lbfgs) lbfgs_losses else NULL,
      final_loss = final_loss
    )
    all_runs[[run_idx]] <- current_result

    # Check if this is the best result
    if (final_loss < best_loss) {
      best_loss <- final_loss
      best_result <- current_result
    }
  }

  # Return the best result and all runs
  return(list(
    best_result = best_result,
    all_runs = all_runs
  ))
}


#' Generate Non-Gaussian Random Variables
#'
#' This function generates non-Gaussian random variables of specified type.
#'
#' @param n An integer specifying the number of observations to generate.
#' @param type A character string specifying the type of non-Gaussian distribution. One of "skew_positive", "skew_negative", or "kurtotic".
#' @return A numeric vector of length \code{n} with non-Gaussian random variables.
#' @export
#' @examples
#' x <- generate_non_gaussian(1000, type = "skew_positive")
generate_non_gaussian <- function(n, type = c("skew_positive", "skew_negative", "kurtotic")) {
  type <- match.arg(type)
  if (type == "skew_positive") {
    return(rgamma(n, 2, runif(1,1,6))) # Positively skewed distribution
  } else if (type == "skew_negative") {
    return(-rgamma(n, 2, runif(1,1,6))) # Negatively skewed distribution
  } else if (type == "kurtotic") {
    return(rt(n, df = runif(1,6,8))) # Kurtotic distribution (heavy-tailed t-distribution)
  }
}

#' Generate a Matrix of Non-Gaussian Variables
#'
#' This function generates a matrix where each column is a non-Gaussian random variable of a specified type.
#'
#' @param n An integer specifying the number of observations (rows).
#' @param k An integer specifying the number of variables (columns).
#' @return A numeric matrix of size \code{n x k} with non-Gaussian variables.
#' @export
#' @examples
#' z <- generate_matrix(1000, 5)
generate_matrix <- function(n, k) {
  types <- c("skew_positive", "skew_negative", "kurtotic")

  # Initialize an empty matrix
  matrix_data <- matrix(NA, nrow = n, ncol = k)

  # Cycle through the different types of non-Gaussian distributions for each column
  for (i in 1:k) {
    matrix_data[, i] <- scale(generate_non_gaussian(n, types[(i - 1) %% length(types) + 1]))
  }

  return(matrix_data)
}
