## OverICA

the R package `OverICA` estiamtes overcomplete indepedent components, meaning it estimates more altent variables then there are observed variables in the data. We observe p variabes in matrix y, we estimate the matrix A that contains the effects of k (non-gaussian) latent variables in x on the observed variables. there are potentially more variables k then p. 

y = Ax

this code estimates A by levraging the differences betwene the observed generalized covariance matrices and means in the data and the model implied generalized covariance matrices and means. This is based on ideas developed by Podosinnikova et al. (2019). 


Unlike Podosinnikova et al. I use back propagation to estimate the parameters. Based on ideas in Ging et al (2019) I define a generative neura network for each of the k latent variables (a multi=layer perceptron), and a matrix A that mixed these variables into p observed pseudo variables. I train the model to approzimate the the generalized covariacne matrices of the obersed data. Unlike Deng et al. I explicitly penalize the loss to ensure the latent variables remain uncorrelated.

**the key gain over other overcomplete ICA techniques is** that we only use 2nd order statistics, no skewness and kurtosis related math needed! Which is great because higher order moments like skewness and kuertosis, or higher order cumulants usually means high dimensional opimization. 

# installation

Before proceeding, ensure that you have the necessary dependencies installed. You can install the `OverICA` package from your local directory or repository once it's built. Additionally, install the required packages if you haven't already:

```R
# Install necessary packages
install.packages(c("torch", "clue", "MASS", "devtools"))

# Install OverICA package (replace 'path/to/package' with your actual path)
# devtools::install_github("MichelNivard/OverICA")
```

## Generic usage

Here is a generic call to the `overica()` function the data in a n by p matrix (n ob ad p variables), k is the number of components, pick k below $\frac{p^2}{4}$. **be aware** this is a sort of neural network (see scientific background below) undertraining and relying on suboptimal or generic setings will give you bad results, but no warnings! You will extract some results but you won't know their bad!

```R
# Call the estimation function
result <- overica(
  data = data,
  k = k,
  n_batch = 4096,
  num_t_vals = 12,
  tbound = 0.2,
  lambda = 0,
  sigma = 3,
  hidden_size = 10,
  use_adam = TRUE,
  use_lbfgs = FALSE,
  adam_epochs = 8000,
  adam_lr = 0.1,
  lbfgs_epochs = 45,
  lr_decay = 0.999
)

```



# Scientific Background: Understanding the Covariance Matrix via the Cumulant Generating Function

In statistical analysis and parameter estimation, particularly within the `OverICA` package, it's crucial to comprehend the relationship between the **covariance matrix** and the **Cumulant Generating Function (CGF)**. This understanding underpins accurate parameter estimation and model fitting.

## What are Cumulants?

**Cumulants** are statistical measures that provide insights into the shape and structure of a probability distribution. Unlike moments (e.g., mean, variance), cumulants have properties that make them particularly useful for analyzing dependencies and higher-order interactions.

- **First Cumulant ($\kappa_1$)**: Mean ($\mu$)
- **Second Cumulant ($\kappa_2$)**: Variance ($\sigma^2$) for univariate distributions; **Covariance ($\Sigma$)** for multivariate distributions
- **Higher-Order Cumulants**: Related to skewness, kurtosis, etc.

## Cumulant Generating Function (CGF)

The **Cumulant Generating Function (CGF)** encapsulates all cumulants of a random variable or vector. For a random vector $\mathbf{X} = (X_1, X_2, \ldots, X_p)^T$, the CGF is defined as:

$$
K_{\mathbf{X}}(\mathbf{t}) = \log \mathbb{E}\left[ e^{\mathbf{t}^T \mathbf{X}} \right]
$$

where $\mathbf{t} = (t_1, t_2, \ldots, t_p)^T$ is a vector of real numbers.

### Key Properties

1. **Cumulants via Derivatives**: The cumulants are obtained by taking partial derivatives of the CGF evaluated at $\mathbf{t} = \mathbf{0}$.
2. **Additivity for Independence**: If $\mathbf{X}$ and $\mathbf{Y}$ are independent, then $K_{\mathbf{X} + \mathbf{Y}}(\mathbf{t}) = K_{\mathbf{X}}(\mathbf{t}) + K_{\mathbf{Y}}(\mathbf{t})$.

## Covariance Matrix from CGF

The **covariance matrix** $\Sigma$ captures the pairwise linear relationships between components of $\mathbf{X}$ and is directly derived from the second-order cumulants of the CGF.

### Derivation

1. **First Derivatives (Means)**:
   
   $\left. \frac{\partial K_{\mathbf{X}}(\mathbf{t})}{\partial t_i} \right|_{\mathbf{t}=0} = \mathbb{E}[X_i] = \mu_i$

2. **Second Derivatives (Covariances)**:
   
 $\left. \frac{\partial^2 K_{\mathbf{X}}(\mathbf{t})}{\partial t_i \partial t_j}\right|{\mathbf{t}=0} = \text{Cov}(X_i, X_j) = \Sigma_{ij}$
 
Thus, the covariance matrix is formed by the second-order partial derivatives of the CGF evaluated at zero:

$$
\Sigma = \begin{pmatrix}
\Sigma_{11} & \Sigma_{12} & \dots & \Sigma_{1p} \\
\Sigma_{21} & \Sigma_{22} & \dots & \Sigma_{2p} \\
\vdots & \vdots & \ddots & \vdots \\
\Sigma_{p1} & \Sigma_{p2} & \dots & \Sigma_{pp} \\
\end{pmatrix}
$$

Each element $\Sigma_{ij}$ represents the covariance between $X_i$ and $X_j$, derived from the second-order cumulants.


## Generalized covariances

So we can define the covariances and emans in terms of the cumulant generating fuction (EGF) evaluated at t =0. But a key insight in **Podosinnikova et al.** and previous work is that you can evaluate the cumulant generating functino at other values of $t$ to get additional information on the distribution of the data. These other evalautinos of the CGF can be viewed as generalizedcovariance mtrices:

 $\left. \frac{\partial^2 K_{\mathbf{X}}(\mathbf{t})}{\partial t_i \partial t_j}\right|{\mathbf{t}!=0} = \text{GenCov}(X_i, X_j) = \Sigma_{ij}$ 





in `OverICA` we evaluate the emperical cumulant generating fucntion of the data at a number of points, and we train a model to generate data that matches the emperical data at these points, which means it has to learn the loadigns in matrix $A$.





## Generative adverserial networks.
 
We base our inference on **Ding et al.** A generative advrserial network is a neural network that "learns" to approximate a data distribution. In outr case the network will learn to mimic the generalized covariances of the actual data. The process is depicted in the diagram below:

![image](https://github.com/user-attachments/assets/56be065f-bbd5-4877-9f71-ca0ba9633c56)

We start with 1 sample of random normally distributed data vectors ($z$)  and we train a neural entwork ( a multi layer perceptron) which is just an expensive workd for a very flxible non-linear transformation. which transforms $z_i$ into $s_i$. Unlike **Ding et al.** we implement a constraint that ensures the variables $s$ are uncorrelated and standardized (constraining their covariance to an identity matrix). the variabes s are  related to the simulated observed variables with the ICA formula:

$$\hat{y} = As$$

And the generalized covariances of the pseudo data $\hat{y}* are matched to the true data $y$ trough optimisation in torch. 

## Inferences details
We arent interested in learning *specific* generalizedcovariances of the data, so for each itteratino fo the optimizer we ample fresh values t and evaluete the emperical and model implied generalized covriances. this ensures we dont accidentally overfit to a specific poorly estimated set of emperical generalized covariances. 



**References:**



Podosinnikova, A., Perry, A., Wein, A. S., Bach, F., d’Aspremont, A., & Sontag, D. (2019, April). Overcomplete independent component analysis via SDP. In The 22nd international conference on artificial intelligence and statistics (pp. 2583-2592). PMLR.

Ding, C., Gong, M., Zhang, K., & Tao, D. (2019). Likelihood-free overcomplete ICA and applications in causal discovery. Advances in neural information processing systems, 32.
