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

#' Compute the Empirical Cumulant Generating Function (ECGF) Using Torch
#'
#' This internal function computes the ECGF and its first and second derivatives using torch tensors.
#'
#' @param t_vector A torch tensor representing the t-vector (p-dimensional).
#' @param data_tensor A torch tensor representing the data (n x p).
#' @return A list containing \code{K} (scalar), \code{dK} (first derivative), and \code{d2K} (second derivative).
#' @export
torch_ecgf <- function(t_vector, data_tensor) {
  # Compute the dot product between data and t_vector
  dot_product <- data_tensor$matmul(t_vector)

  # Subtract max for numerical stability
  max_dot <- torch_max(dot_product)
  dot_product <- dot_product - max_dot

  # Compute the exponential of the dot product
  exp_dot <- torch_exp(dot_product)

  # Compute K(t) = log(mean(exp(dot_product))) + max_dot
  K_t <- torch_log(torch_mean(exp_dot)) + max_dot

  # Compute weights for the weighted mean
  weights <- exp_dot / torch_sum(exp_dot)

  # Compute the first derivative dK = E[data * weights]
  dK <- torch_sum(data_tensor * weights$unsqueeze(2), dim = 1)

  # Compute the second derivative d2K
  data_centered <- data_tensor - dK
  d2K <- (weights$unsqueeze(2) * data_centered)$t()$matmul(data_centered)

  return(list(K = K_t, dK = dK, d2K = d2K))
}

#' Estimate an Overcomplete Independent Component Analysis (OICA) Using Torch and Empirical Cumulant Generating Functions
#'
#' This function estimates parameters of an OICA model using torch tensors and ECGFs. It optimizes the parameters using a combination of Adam and L-BFGS optimizers.
#'
#' @param data A numeric matrix representing the observed data (n x p).
#' @param k An integer specifying the number of latent variables.
#' @param num_t_vals An integer specifying the number of t-values to use (default is 7).
#' @param lambda A numeric value for the L1 regularization parameter (default is 0.01).
#' @param sigma A numeric value for the covariance penalty parameter (default is 0.01).
#' @param tbound A numeric value specifying the bound for t-values (default is 0.4).
#' @param n_batch An integer specifying the batch size for the neural network (default is 1024).
#' @param hidden_size An integer specifying the hidden size of the neural network (default is 10).
#' @param use_adam Logical; whether to use the Adam optimizer first (default is TRUE).
#' @param adam_epochs An integer specifying the number of epochs for Adam optimizer (default is 100).
#' @param adam_lr A numeric value specifying the learning rate for Adam optimizer (default is 0.01).
#' @param use_lbfgs Logical; whether to use the L-BFGS optimizer after Adam (default is TRUE).
#' @param lbfgs_epochs An integer specifying the number of epochs for L-BFGS optimizer (default is 50).
#' @param lbfgs_lr A numeric value specifying the learning rate for L-BFGS optimizer (default is 1).
#' @param lbfgs_max_iter An integer specifying the max iterations per optimization step for L-BFGS (default is 20).
#' @param lr_decay A numeric value specifying the decay rate for the multiplicative learning rate decay; set to 1 for no decay (default is 0.999).
#' @return A list containing the estimated A matrix (\code{A_est}) and covariance matrix (\code{cov}).
#' @import torch
#' @export
#' @examples
#' # Assuming 'data' is a matrix of observed data
#' result <- estimate_parameters(data, k = 5)
overica <- function(
    data,
    k,
    num_t_vals = 7,
    lambda = 0.01,
    sigma=0.01,
    tbound = .4,
    n_batch = 1024,
    hidden_size = 10,
    use_adam = TRUE,       # Whether to use Adam optimizer first
    adam_epochs = 100,     # Number of epochs for Adam optimizer
    adam_lr = 0.01,        # Learning rate for Adam optimizer
    use_lbfgs = TRUE,      # Whether to use L-BFGS optimizer after Adam
    lbfgs_epochs = 50,     # Number of epochs for L-BFGS optimizer
    lbfgs_lr = 1,          # Learning rate for L-BFGS optimizer
    lbfgs_max_iter = 20,    # Max iterations per optimization step for L-BFGS
    lr_decay = 0.999       # decay rate for multiplicative learning rate decay, set to 1 for no decay
) {
  # Check if CUDA is available (for GPU acceleration)
  device <- if (cuda_is_available()) torch_device("cuda") else torch_device("cpu")

  # Data tensor
  data_tensor <- torch_tensor(data, dtype = torch_float(), device = device)

  n <- nrow(data)
  p <- ncol(data)

  # Initialize parameters with gradients enabled
  # A matrix parameters (flattened, requires_grad = TRUE)
  A_params <- torch_tensor(runif(p * k, min = -0.5, max = 0.5), requires_grad = TRUE, device = device)

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
        s_i <- self$nets[[i]](z_i)
        # Centering and scaling
        s_mean <- s_i$mean()
        s_std <- s_i$std()
        s_scaled <- (s_i - s_mean) / s_std
        s_list[[i]] <- s_scaled
      }
      s <- torch_cat(s_list, dim = 2)
      return(s)
    }
  )

  neural_nets <- NeuralNets(k = k, hidden_size = hidden_size)$to(device = device)

  # List of parameters to optimize
  params <- c(list(A_params), neural_nets$parameters)

  # Sample z ~ N(0,I) once and keep it fixed
  z_fixed <- torch_randn(c(n_batch, k), device = device)

  # Covariance penalty function
  covariance_penalty <- function(s) {
    n <- s$size(1)  # Batch size

    # Compute covariance matrix
    cov_matrix <- (s$t()$matmul(s)) / (n - 1)

    # Penalize off-diagonal elements (i.e., correlations)
    cov_penalty <- torch_sum(torch_square(cov_matrix - torch_eye(s$size(2), device = device)))

    return(cov_penalty)
  }

  # Covariance matrix
  covariance_matrix <- function(s) {
    n <- s$size(1)  # Batch size

    # Compute covariance matrix
    cov_matrix <- (s$t()$matmul(s)) / (n - 1)
    return(cov_matrix)
  }

  # Define the objective function using torch operations
  fn <- function() {
    # Reshape A_params into a p x k matrix
    A <- A_params$view(c(p, k))

    # Use fixed z
    z <- z_fixed

    # Pass z through neural_nets to get s
    s <- neural_nets(z)

    # Compute data_hat = s %*% t(A)
    data_hat <- s$matmul(A$t())

    # Initialize loss
    loss <- torch_tensor(0.0, device = device)

    # Resample t_vals_o in each function call
    # Define t-values for observed variables (t_vals_o)
    # The first t-value is a zero vector
    t_vals_o <- matrix(0, nrow = 1, ncol = p)
    # Additional random t-values
    additional_t_vals <- matrix(runif((num_t_vals - 1) * p, min = -tbound, max = tbound), ncol = p)
    t_vals_o <- rbind(t_vals_o, additional_t_vals)
    t_vals_o_tensor <- torch_tensor(t_vals_o, dtype = torch_float(), device = device)

    # Compute data_dK and data_d2K using batched torch_ecgf
    data_dK_list <- list()
    data_d2K_list <- list()
    for (i in 1:num_t_vals) {
      t_vector <- t_vals_o_tensor[i, ]
      ecgf_result <- torch_ecgf(t_vector, data_tensor)
      data_dK_i <- ecgf_result$dK  # (p)
      data_dK_list[[i]] <- data_dK_i$detach()
      data_d2K_i <- ecgf_result$d2K  # (p x p)
      data_d2K_list[[i]] <- data_d2K_i$detach()
    }

    # Loop over each t_vector in t_vals_o_tensor
    for (i in 1:num_t_vals) {
      t_vector <- t_vals_o_tensor[i, ]

      # Compute model's K(t), dK(t), d2K(t) using data_hat
      ecgf_result_model <- torch_ecgf(t_vector, data_hat)
      model_dK_i <- ecgf_result_model$dK  # (p)
      model_d2K_i <- ecgf_result_model$d2K  # (p x p)

      # Retrieve precomputed data_dK_i and data_d2K_i
      data_dK_i <- data_dK_list[[i]]  # (p)
      data_d2K_i <- data_d2K_list[[i]]  # (p x p)

      # Compute the difference in dK
      diff_dK <- model_dK_i - data_dK_i
      # Compute the squared L2 norm of the difference
      loss_dK <- torch_sum(diff_dK * diff_dK)

      # Compute the difference in d2K
      diff_d2K <- model_d2K_i - data_d2K_i
      # Compute the squared Frobenius norm of the difference
      loss_d2K <- torch_sum(diff_d2K * diff_d2K)

      # Accumulate the losses
      loss <- loss + loss_dK + loss_d2K
    }

    # Add L1 regularization term for A, covariance penalty for s
    loss <- loss + lambda * torch_sum(torch_abs(A)) + sigma * covariance_penalty(s)

    return(loss)
  }

  # Run Adam optimizer first if specified
  if (use_adam) {
    optimizer_adam <- optim_adam(params = params, lr = adam_lr)

    # Define learning rate scheduler for Adam
    scheduler <- lr_multiplicative(optimizer_adam, lr_lambda = function(epoch) lr_decay)

    for (epoch in 1:adam_epochs) {
      optimizer_adam$zero_grad()
      loss_value <- fn()
      loss_value$backward()
      optimizer_adam$step()

      # Step the scheduler to update the learning rate
      scheduler$step()

      # Optional: Print progress
      if (epoch %% 10 == 0) {
        current_lr <- optimizer_adam$param_groups[[1]]$lr
        cat("\r", "Adam Epoch:", epoch, "Loss:", as.numeric(loss_value$item()), "LR:", current_lr)
      }
      if (epoch == adam_epochs) {
        cat("\n", "Final Adam Epoch:", epoch, "Loss:", as.numeric(loss_value$item()), "\n")
      }
    }
  }

  # Continue with L-BFGS optimizer if specified
  if (use_lbfgs) {
    optimizer_lbfgs <- optim_lbfgs(
      params = params,
      lr = lbfgs_lr,
      max_iter = lbfgs_max_iter,
      history_size = 10,
      line_search_fn = "strong_wolfe"
    )

    # Define learning rate scheduler for L-BFGS
    scheduler_lbfgs <- lr_multiplicative(optimizer_lbfgs, lr_lambda = function(epoch) lr_decay)

    for (epoch in 1:lbfgs_epochs) {
      # For L-BFGS optimizer, define closure
      closure <- function() {
        optimizer_lbfgs$zero_grad()
        loss <- fn()
        loss$backward()
        return(loss)
      }
      # Perform an optimization step
      loss_value <- optimizer_lbfgs$step(closure)

      # Step the scheduler to update the learning rate
      scheduler_lbfgs$step()

      # Optional: Print progress
      if (epoch %% 1 == 0) {
        current_lr <- optimizer_lbfgs$param_groups[[1]]$lr
        cat("\r", "L-BFGS Epoch:", epoch, "Loss:", as.numeric(loss_value$item()), "LR:", current_lr)
      }
    }
  }

  # Retrieve optimized parameters
  A_est <- A_params$view(c(p, k))

  # Return estimated A and covariance matrix
  return(list(A_est = A_est, cov = covariance_matrix(neural_nets(z_fixed))))
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
