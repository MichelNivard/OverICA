

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
overica.gcov <- function(
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



#' OverICA with Structural (I - B)^{-1}, Overcomplete Latent s, and Optional Error. Estimation based on higher order moments.
#'
#' This function implements a model:
#'   data_hat = (I - B)^(-1) (s %*% A) + e
#'
#' where:
#'   - B is a p x p matrix of direct causal/structural effects among the p obs. variables.
#'   - A is a p x k mixing matrix for the k latent sources s.
#'   - s is produced by a small neural net from z ~ Normal(0,I).
#'   - e ~ MVN(0, error_cov) is optional known noise (if error_cov is non-NULL).
#'
#' We estimate B, A, and the net's parameters by matching the user-supplied "moment_func"
#' of data_hat to that of the observed data. Typically, you'd use e.g.
#' torch_unique_4th_central_moments() or torch_all_4th_central_moments() for 4th-order stats,
#' but you can supply any function returning a 1D torch vector for the mismatch.
#'
#' @param data A numeric matrix of shape (n, p) for observed data.
#' @param k Number of latent sources s.
#' @param moment_func A function that takes a (n_batch x p) torch tensor and returns
#'   a 1D torch tensor of statistics to match. (e.g. `torch_unique_4th_central_moments`).
#' @param third A logical the specifies whether third order moments are to be considered in the loss
#' @param error_cov An optional (p x p) covariance for Gaussian noise e. If NULL, no error added.
#' @param maskB Optional p x p binary mask for B. 1 => estimate the entry, 0 => fix to 0.
#' @param maskA Optional p x k binary mask for A. 1 => estimate, 0 => fix to 0.
#' @param lambdaA L1 penalty on A, only penalizes overcomplete latent variables where p < k (i.e leaves the unique residuals untouched).
#' @param lambdaB L1 penalty on B.
#' @param sigma Covariance penalty weight for the sources s (to encourage whitening).
#' @param hidden_size Number of hidden units in the small MLP that maps z->s.
#' @param n_batch Batch size for the random z draws.
#' @param use_adam Whether to run Adam first.
#' @param adam_epochs Number of Adam epochs.
#' @param adam_lr Adam learning rate.
#' @param use_lbfgs Whether to run L-BFGS afterwards.
#' @param lbfgs_epochs Number of L-BFGS epochs (outer loop calls).
#' @param lbfgs_lr L-BFGS learning rate.
#' @param lbfgs_max_iter Max iteration per LBFGS step.
#' @param lr_decay Multiplicative LR decay per epoch (1 => no decay).
#' @param clip_grad Whether to clip the gradient norm to 2.
#' @param num_runs How many random restarts to do, picking the best final loss.
#'
#' @return A list with:
#'   \item{best_result}{the run with lowest final loss}
#'   \item{all_runs}{a list of all runs and their final parameters/loss}
#'
#' @import torch
#' @export
overica.moments.sem <- function(
  data,
  k,
  moment_func,
  third=TRUE,
  error_cov = NULL,
  maskB = NULL,
  maskA = NULL,
  lambdaA = 0.01,
  lambdaB = 0.00,
  sigma   = 0.01,
  hidden_size = 10,
  n_batch = 1024,
  use_adam = TRUE,
  adam_epochs = 100,
  adam_lr = 0.01,
  use_lbfgs = TRUE,
  lbfgs_epochs = 50,
  lbfgs_lr = 1,
  lbfgs_max_iter = 20,
  lr_decay = 0.999,
  clip_grad = TRUE,
  num_runs  = 1
) {
# Setup
device <- if (cuda_is_available()) torch_device("cuda") else torch_device("cpu")

data_tensor <- torch_tensor(data, dtype=torch_float(), device=device)
n <- nrow(data)
p <- ncol(data)
  
# Compute indices for moment loss functions
indices <-  compute_unique_moment_indices(p)

# 1) Precompute the observed statistic (4th moments, or user-defined)
obs_stat <- moment_func(data_tensor,indices,third)

# 2) Optional masks
if (is.null(maskB)) maskB <- matrix(1, p, p)
if (is.null(maskA)) maskA <- matrix(1, p, k)
maskB_t <- torch_tensor(maskB, dtype=torch_float(), device=device)$detach()
maskA_t <- torch_tensor(maskA, dtype=torch_float(), device=device)$detach()

# 3) Optional known error
if (!is.null(error_cov)) {
  cov_t <- torch_tensor(error_cov, dtype=torch_float(), device=device)
}

# A small net to produce s from z
NetModule <- nn_module(
  initialize = function(k, hidden_size) {
    self$nets <- nn_module_list()
    for (i in seq_len(k)) {
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
    out_list <- list()
    for (i in seq_len(k)) {
      out_list[[i]] <- self$nets[[i]](z[, i, drop=FALSE])
    }
    torch_cat(out_list, dim=2)  # shape: (n_batch, k)
  }
)

# Cov penalty to encourage s to be white
covariance_penalty <- function(s) {
  ns <- s$size(1)
  cov_s <- (s$t()$matmul(s)) / (ns - 1)
  diff <- cov_s - torch_eye(k, device=device)
  torch_sum(diff * diff)
}

all_runs <- vector("list", num_runs)
best_loss <- Inf
best_result <- NULL

for (run_idx in seq_len(num_runs)) {
  cat(sprintf("\n=== Run %d of %d ===\n", run_idx, num_runs))

  # 4) Initialize B, A, and the net
  B_params <- torch_tensor(
    runif(p*p, min=-0.05, max=0.05),
    requires_grad=TRUE, device=device
  )
  A_params <- torch_tensor(
    runif(p*k, min=-0.2, max=0.2),
    requires_grad=TRUE, device=device
  )

  nnets <- NetModule$new(k, hidden_size)$to(device=device)

  # Combine in param list
  params <- c(list(B_params, A_params), nnets$parameters)

  # 5) Sample z ~ Normal(0,I), and maybe e
  z_fixed <- torch_randn(c(n_batch, k), device=device)
  e_fixed <- NULL
  if (!is.null(error_cov)) {
    e_dist <- distr_multivariate_normal(torch_zeros(p), cov_t)
    e_fixed <- e_dist$sample(n_batch)
  }



  # 6) Define the forward/loss
  fn <- function() {
    B_mat <- B_params$view(c(p, p)) * maskB_t
    A_mat <- A_params$view(c(p, k)) * maskA_t

    # (I - B)
    I_p <- torch_eye(p, device=device)
    IB  <- I_p - B_mat

    # Invert
    # You might want to use a solve: M <- torch_linalg_solve(IB, I_p)
    # or a stable decomposition. We'll do a direct inverse for simplicity:
    M <- torch_inverse(IB)

    # s
    s_val <- nnets(z_fixed)  # (n_batch, k)
  
    s_val <- s_val - s_val$mean()
    # s %*% A => (n_batch, p)
    SA <- s_val$matmul(A_mat$t())

    # data_hat = M %*% SA^T, then transpose => shape (n_batch, p)
    # i.e. each row is M times the row of SA
    data_hat_l <- M$matmul(SA$t())$t()

    if (!is.null(e_fixed)) {
      data_hat <- data_hat_l + e_fixed
    }else{
      data_hat <-  data_hat_l
    }

    # compute model's statistic
    model_stat <- moment_func(data_hat,indices,third)
    diff <- model_stat - obs_stat
    loss_stat <- torch_mean(diff * diff)

    # L1 penalties
    if(p < k){
    loss_l1A <- lambdaA * torch_sum(torch_abs(A_mat[,1:(k-p)]))
    } else{
      loss_l1A <- 0 
    }

    loss_l1B <- lambdaB * torch_sum(torch_abs(B_mat))

    # Cov penalty on s
    loss_cov <- sigma * covariance_penalty(s_val)

    loss <- loss_stat + loss_l1A + loss_l1B + loss_cov
    loss
    }

  # 7) Adam (optional)
  if (use_adam) {
    optimizer_adam <- optim_adam(params, lr=adam_lr)
    scheduler_adam <- lr_multiplicative(
      optimizer_adam,
      lr_lambda=function(epoch) lr_decay
    )
    for (epoch in seq_len(adam_epochs)) {
      optimizer_adam$zero_grad()
      lv <- fn()
      lv$backward()
      if (clip_grad) {
        nn_utils_clip_grad_norm_(params, max_norm=2)
      }
      optimizer_adam$step()
      scheduler_adam$step()
      if (epoch %% 10 == 0 || epoch==adam_epochs) {
        cat(sprintf("   [Adam epoch %d] loss=%.6f\n", epoch, lv$item()))
      }
    }
  }

  # 8) L-BFGS (optional)
  lbfgs_losses <- numeric(0)
  if (use_lbfgs) {
    optimizer_lbfgs <- optim_lbfgs(
      params = params,
      lr = lbfgs_lr,
      max_iter = lbfgs_max_iter,
      history_size = 10,
      line_search_fn = "strong_wolfe"
    )
    scheduler_lbfgs <- lr_multiplicative(
      optimizer_lbfgs,
      lr_lambda=function(epoch) lr_decay
    )
    lbfgs_losses <- numeric(lbfgs_epochs)

    for (epoch in seq_len(lbfgs_epochs)) {
      closure <- function() {
        optimizer_lbfgs$zero_grad()
        lv <- fn()
        lv$backward()
        lv
      }
      lv_val <- optimizer_lbfgs$step(closure)
      scheduler_lbfgs$step()
      lbfgs_losses[epoch] <- as.numeric(lv_val$item())
      cat(sprintf("   [LBFGS epoch %d] loss=%.6f\n", epoch, lbfgs_losses[epoch]))
    }
  }

  # Final
  final_loss <- as.numeric(fn()$item())
   

  # Extract final B, A
  B_est <- (B_params$view(c(p,p)) * maskB_t)
  A_est <- (A_params$view(c(p,k)) * maskA_t)

      # (I - B)
      I_p <- torch_eye(p, device=device)
      IB  <- I_p - B_est
  
      # Invert
      # You might want to use a solve: M <- torch_linalg_solve(IB, I_p)
      # or a stable decomposition. We'll do a direct inverse for simplicity:
      M <- torch_inverse(IB)
  
      # s
      s_val <- nnets(z_fixed)  # (n_batch, k)
    
      s_val <- s_val - s_val$mean()
      # s %*% A => (n_batch, p)
      SA <- s_val$matmul(A_est$t())
  
      # data_hat = M %*% SA^T, then transpose => shape (n_batch, p)
      # i.e. each row is M times the row of SA
      data_hat_l <- M$matmul(SA$t())$t()
      
    if (!is.null(e_fixed)) {
      data_hat <- data_hat_l + e_fixed
    }else{
      data_hat <-  data_hat_l
    }
  
      model_stat <- moment_func(data_hat,indices,third)
      diff <- model_stat - obs_stat
      loss_stat <- torch_mean(diff * diff)
  
  moments <- moment_func(data_hat_l,indices,third)
  out <-  extract_covariance_and_kurtosis(moments,indices,p,third)
  current_result <- list(
    B_est       = B_est,
    A_est       = A_est,
    moments     = out,
    net         = nnets,
    final_loss  = final_loss,
    raw_loss    = loss_stat,
    lbfgs_losses= lbfgs_losses
  )
  all_runs[[run_idx]] <- current_result
  if (final_loss < best_loss) {
    best_loss   <- final_loss
    best_result <- current_result
  }
} # end for run_idx

list(best_result=best_result, all_runs=all_runs)
}

#' Grid Search over L1 Penalties for OverICA SEM
#'
#' This function performs a grid search over the L1 penalty hyperparameters for matrices A and B
#' in an OverICA SEM model. The underlying estimation function (e.g. \code{overica.moments.sem})
#' is called repeatedly with candidate penalty values. In the first iteration, candidate values
#' span one order of magnitude up and down from the initial values. In subsequent iterations,
#' the grid is refined using multiplicative factors of 0.5, 1, and 2 around the current best.
#' The candidate pair with the lowest final loss is selected and used as the center for the next iteration.
#'
#' @param data A numeric matrix of observed data (n x p).
#' @param k Number of latent sources.
#' @param moment_func A function that takes a torch tensor (n_batch x p), along with moment indices
#'        and a logical flag for third order moments, and returns a 1D torch tensor of computed moments.
#' @param error_cov Optional (p x p) covariance matrix for noise. If NULL, no error is added.
#' @param maskB Optional binary mask (p x p) for matrix B. A 1 indicates the entry is estimated,
#'        and a 0 forces the entry to 0.
#' @param maskA Optional binary mask (p x k) for matrix A. A 1 indicates the entry is estimated,
#'        and a 0 forces the entry to 0.
#' @param sigma Covariance penalty weight for the latent sources s (to encourage whitening).
#' @param third Do we consider the third moments, dont use if your distributions are suposed to be symetric. 
#' @param hidden_size Number of hidden units in the neural network that maps z to s.
#' @param n_batch Batch size for the random z draws.
#' @param use_adam Logical; whether to run the Adam optimizer.
#' @param adam_epochs Number of epochs for Adam.
#' @param adam_lr Learning rate for Adam.
#' @param use_lbfgs Logical; whether to run the L-BFGS optimizer after Adam.
#' @param lbfgs_epochs Number of epochs for L-BFGS.
#' @param lbfgs_lr Learning rate for L-BFGS.
#' @param lbfgs_max_iter Maximum iterations per L-BFGS step.
#' @param lr_decay Multiplicative learning rate decay per epoch (set to 1 for no decay).
#' @param clip_grad Logical; whether to clip gradients.
#' @param num_runs Number of random restarts for the estimation.
#' @param initial_lambdaA Initial L1 penalty for matrix A.
#' @param initial_lambdaB Initial L1 penalty for matrix B.
#' @param num_iter Number of grid search iterations.
#'
#' @return A list containing:
#' \describe{
#'   \item{best_result}{The best estimation result, including the final matrices A and B and the final loss.}
#'   \item{best_lambdaA}{The best L1 penalty chosen for matrix A.}
#'   \item{best_lambdaB}{The best L1 penalty chosen for matrix B.}
#'   \item{grid_losses}{A matrix of final losses from the last iteration of the grid search.}
#' }
#'
#' @examples
#' \dontrun{
#'   # Assume 'data' is your observed data and 'torch_unique_24th_central_moments' is your moment function.
#'   grid_res <- grid_search_overica(data = data, k = 5,
#'                     moment_func = torch_central_moments,
#'                     initial_lambdaA = 0.01, initial_lambdaB = 0.00,
#'                     num_iter = 3)
#'   cat("Best lambdaA:", grid_res$best_lambdaA, "\n")
#'   cat("Best lambdaB:", grid_res$best_lambdaB, "\n")
#' }
#'
#' @export
grid_search_overica <- function(data, k, moment_func, error_cov = NULL, maskB = NULL, maskA = NULL,
                                sigma = 0.01, third=TRUE, hidden_size = 10, n_batch = 1024,
                                use_adam = TRUE, adam_epochs = 100, adam_lr = 0.01,
                                use_lbfgs = TRUE, lbfgs_epochs = 50, lbfgs_lr = 1, lbfgs_max_iter = 20,
                                lr_decay = 0.999, clip_grad = TRUE, num_runs = 1,
                                initial_lambdaA = 0.01, initial_lambdaB = 0.00,
                                num_iter = 3,threshold = 0.001,crit="AIC") {
  device <- if (cuda_is_available()) torch_device("cuda") else torch_device("cpu")
  n <- nrow(data)
  p <- ncol(data)
  
  best_lambdaA <- initial_lambdaA
  best_lambdaB <- initial_lambdaB
  best_loss <- Inf
  best_result <- NULL
  
  # For the first iteration, search over one order of magnitude; then refine.
  for (iter in seq_len(num_iter)) {

      grid_factors <-  exp((1/iter) * log(c(0.25, 1, 4)))

    
    candidate_lambdaA <- best_lambdaA * grid_factors
    candidate_lambdaB <- best_lambdaB * grid_factors
    
    grid_losses <- matrix(NA, nrow = length(candidate_lambdaA), ncol = length(candidate_lambdaB))
    grid_results <- list()
    
    for (i in seq_along(candidate_lambdaA)) {
      for (j in seq_along(candidate_lambdaB)) {
        cat(sprintf("Trying lambdaA = %.5f, lambdaB = %.5f\n",
                    candidate_lambdaA[i], candidate_lambdaB[j]))
        
        res <- overica.moments.sem(data = data, k = k, moment_func = moment_func,
                                   third = third, error_cov = error_cov,
                                   maskB = maskB, maskA = maskA,
                                   lambdaA = candidate_lambdaA[i],
                                   lambdaB = candidate_lambdaB[j],
                                   sigma = sigma, hidden_size = hidden_size,
                                   n_batch = n_batch,
                                   use_adam = use_adam, adam_epochs = adam_epochs, adam_lr = adam_lr,
                                   use_lbfgs = use_lbfgs, lbfgs_epochs = lbfgs_epochs, lbfgs_lr = lbfgs_lr,
                                   lbfgs_max_iter = lbfgs_max_iter, lr_decay = lr_decay,
                                   clip_grad = clip_grad, num_runs = num_runs)
        loss_here <- as.numeric(res$best_result$raw_loss)
        n_usable <- n
        if(crit== "AIC"){
        aic_val <- compute_AIC(loss_here, res$best_result$A_est, res$best_result$B_est,k=k, n_used = n_usable,threshold = threshold)$AIC
        cat("Iteration completed with AIC:", aic_val,"\n")
        grid_losses[i, j] <- aic_val
        }
        if(crit== "BIC"){
          bic_val <- compute_BIC(loss_here, res$best_result$A_est, res$best_result$B_est,k=k, n_used = n_usable,threshold = threshold)$BIC
          cat("Iteration completed with BIC:", bic_val,"\n")
          grid_losses[i, j] <- bic_val
          }
        grid_results[[paste(i, j, sep = "_")]] <- res$best_result
        }
    }
    
    best_idx <- which(grid_losses == min(grid_losses), arr.ind = TRUE)
    best_i <- best_idx[1, 1]
    best_j <- best_idx[1, 2]
    best_loss <- grid_losses[best_i, best_j]
    best_lambdaA <- candidate_lambdaA[best_i]
    best_lambdaB <- candidate_lambdaB[best_j]
    best_result <- grid_results[[paste(best_i, best_j, sep = "_")]]
    
    cat(sprintf("Iteration %d best: lambdaA = %.5f, lambdaB = %.5f with loss = %.5f\n",
                iter, best_lambdaA, best_lambdaB, best_loss))
  }
  
  return(list(best_result = best_result,
              best_lambdaA = best_lambdaA,
              best_lambdaB = best_lambdaB,
              grid_losses = grid_losses))
}
