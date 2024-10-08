% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/functions.R
\name{overica}
\alias{overica}
\title{Estimate an Overcomplete Independent Component Analysis (OICA) Using Torch and Empirical Cumulant Generating Functions}
\usage{
overica(
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
  use_adam = TRUE,
  adam_epochs = 100,
  adam_lr = 0.01,
  use_lbfgs = TRUE,
  lbfgs_epochs = 50,
  lbfgs_lr = 1,
  lbfgs_max_iter = 20,
  lr_decay = 0.999,
  num_runs = 1
)
}
\arguments{
\item{data}{A numeric matrix representing the observed data (n x p).}

\item{k}{An integer specifying the number of latent variables.}

\item{num_t_vals}{An integer specifying the number of t-values to use (default is 8).}

\item{lambda}{A numeric value for the L1 regularization parameter (default is 0.01).}

\item{sigma}{A numeric value for the covariance penalty parameter (default is 0.01).}

\item{tbound}{A numeric value specifying the standard deviation for t-values (default is 0.2).}

\item{abound}{A numeric value that specifies the upper and lower bound of the uniformly sampled starting values for A (default is 0.5).}

\item{n_batch}{An integer specifying the batch size for the neural network (default is 1024).}

\item{hidden_size}{An integer specifying the hidden size of the neural network (default is 10).}

\item{neural_nets}{User specified or previously fitted neural nets; if left at NULL, the model will initialize new untrained neural networks.}

\item{clip_grad}{Should the gradient be clipped during optimization?}

\item{use_adam}{Logical; whether to use the Adam optimizer first (default is TRUE).}

\item{adam_epochs}{An integer specifying the number of epochs for Adam optimizer (default is 100).}

\item{adam_lr}{A numeric value specifying the learning rate for Adam optimizer (default is 0.01).}

\item{use_lbfgs}{Logical; whether to use the L-BFGS optimizer after Adam (default is TRUE).}

\item{lbfgs_epochs}{An integer specifying the number of epochs for L-BFGS optimizer (default is 50).}

\item{lbfgs_lr}{A numeric value specifying the learning rate for L-BFGS optimizer (default is 1).}

\item{lbfgs_max_iter}{An integer specifying the max iterations per optimization step for L-BFGS (default is 20).}

\item{lr_decay}{A numeric value specifying the decay rate for the multiplicative learning rate decay; set to 1 for no decay (default is 0.999).}

\item{num_runs}{An integer specifying the number of times to run the estimation (default is 1).}
}
\value{
A list containing the best result (\code{best_result}) and a list of all runs (\code{all_runs}).
}
\description{
This function estimates parameters of an OICA model using torch tensors and ECGFs. It optimizes the parameters using a combination of Adam and L-BFGS optimizers.
}
\examples{
# Assuming 'data' is a matrix of observed data
result <- overica(data, k = 5, num_runs = 10)
}
