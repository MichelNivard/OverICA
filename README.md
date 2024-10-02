## OverICA

**EXPERIMENTAL!!**

the R package `OverICA` performs overcomplete indepedent component analysis (ICA), meaning it estimates the relation between latent variables andobserved variables in the data, allowing there to be more latent then observed variables. We observe p variabes in matrix $y_{n,p}$, we estimate the matrix $A_{p,k}$ that contains the effects of k (non-gaussian) latent variables in $x_{n,k}$ on the observed variables. there are potentially more variables k then p. 

$$y = Ax^t$$

this code estimates A by levraging the differences betweee the observed generalized covariance matrices and generalized means in the data and the model implied generalized covariance matrices and generalized means. This is based on ideas developed by Podosinnikova et al. (2019). 

**We assume:**

1. variables $x_i$  are non-gaussian (skewed, kurtotic sparse etc)
2. variables $x_i$ are uncorrelated
3. a single layer neural network applied to a gaussian variable can approximate the (moments of) the uncorrelated latent variables


Unlike Podosinnikova et al. I use backpropagation to estimate the parameters. Based on ideas in Ding et al (2019) I define a generative neural network for each of the k latent variables (a multi-layer perceptron), and a matrix A that mixed these variables into p observed pseudo variables. I train the model to approzimate the the generalized covariacne matrices of the obersed data. Unlike Deng et al. I explicitly penalize the loss to ensure the latent variables remain uncorrelated.

**the key gain over other overcomplete ICA techniques is** that we only use 2nd order statistics, no skewness and kurtosis related math needed! Which is great because higher order moments like skewness and kurtosis, or higher order cumulants usually means high dimensional matrices and slow opimization. 

**Warning:**


Inferences is aproximate! Here is an example concordance between the true and estimated loadings for a problem with 20 observed and 50 latent variables. I in this case didn't try very hard to optimize the training and parameters of the model to squeeze out maximum concordance, so you might epxect hgher concordance, but take to heart that these loadings are estimates not mathematical properties of the data like svd/eigen values.

![image](https://github.com/user-attachments/assets/02578321-bed7-466a-bf6c-044642bd89af)


# installation

Before proceeding, ensure that you have the necessary dependencies installed. You can install the `OverICA` package from your local directory or repository once it's built. Additionally, install the required packages if you haven't already:

```R
# Install necessary packages
install.packages(c("torch", "clue", "MASS", "devtools"))

# Install OverICA package (replace 'path/to/package' with your actual path)
devtools::install_github("MichelNivard/OverICA")
library(OverICA)
```

## Generic usage

Here is a generic call to the `overica()` function the data in a n by p matrix (n ob ad p variables), k is the number of components, pick k below $\frac{p^2}{4}$.

**be aware** this function fits a custom neural network (see scientific background below) undertraining and relying on suboptimal or generic setings will give you bad results, but no warnings! You will always extract some results but you won't know of they are bad!

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



# Scientific Background: 

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


## Generalized covariances & means

So we can define the covariances and means in terms of the cumulant generating fuction (EGF) evaluated at $t = 0$. But a key insight in **Podosinnikova et al.** and previous work is that you can evaluate the cumulant generating function at other values of $t$ ($t \neq 0$) to get additional information on the distribution of the data. 

These other evaluations of the CGF can be viewed as generalized covariance matrices:

 $\left. \frac{\partial^2 K_{\mathbf{X}}(\mathbf{t})}{\partial t_i \partial t_j}\right|{\mathbf{t}\neq 0} = \text{GenCov}(X_i, X_j) = \Sigma(t)_{ij}$ 

Similarly the generalized mean the the first derivative of the CGF evaluated at $t$. 

In `OverICA` we evaluate the emperical CGF of the data at a number of points ($t$), and we train a model to generate data that matches the emperical data at these points. The model is structured linke an ICA model.


## Generative adverserial networks
 
We base our inference on **Ding et al.** A generative adverserial network is a neural network that "learns" to approximate a data distribution. In our case the network will learn to mimic the generalized covariances of the actual data. The model is depicted in the diagram below:

![image](https://github.com/user-attachments/assets/56be065f-bbd5-4877-9f71-ca0ba9633c56)

We start with 1 sample of random normally distributed data vectors ($z$)  and we train a neural nettwork ( a multi layer perceptron) which in our case is just a posh and very flexible generic non-linear transformation. the neural network transforms $z_i$ into $s_i$. Unlike **Ding et al.**, who pioneered this idea, we implement a constraint that ensures the variables $s$ are uncorrelated and standardized (constraining their covariance to an identity matrix). The variabes s are related to the simulated observed variables via the ICA formula:

$$\hat{y} = As$$

And the generalized covariances of the pseudo data $\hat{y}$ are matched to the true data $y$ trough optimisation in torch. 

So in our moodel the free parameters are:

1. a neural network for each latent variable
2. the loadings matrix **$A$**

Under the following constraint:

a.  variables $s$ are unorrelated
b.  optinally a sparsity constraint on $A$

Mimimizing a function that is a sum of these terms (and any penalties):

1. $$||GenCov(\hat{y}) - GenCov(y)||_2$$
2. $$||GenMean(\hat{y}) - GenMean(y)||_2$$

## Inferences details

We arent interested in learning *specific* generalized covariances of the data, so for each itteration of the optimizer we sample fresh values t and evaluate the emperical and model implied generalized covriances. This ensures we dont accidentally overfit to a specific poorly estimated set of emperical generalized covariances of the data.

1. Sample
-   Sample new values $t$
-   Compute the generalized covariances for the data at $t$
4. Forward Pass (Prediction):
-   Input Data: Feed the input data into the model.
-   Compute Output: The model processes the input through its layers to produce an output (prediction) of the generalized covariances at $t$.
4.  Loss Computation:
-  Calculate Loss: Compute the difference between the predicted outputs and the actual outputs.
5.  Backward Pass (Backpropagation):
-  Compute Gradients: Calculate how much each parameter (weight and bias) contributes to the loss using calculus (derivatives).
-  Gradient Descent: Use these gradients to determine the direction to adjust the parameters to minimize the loss.
6.  Parameter Update:
-  Adjust Parameters: Update the model's parameters slightly in the direction that reduces the loss.


## Other tools:

- [https://github.com/dingchenwei/Likelihood-free_OICA] Likelihood Free Overcomplete ICA
- [https://github.com/gilgarmish/oica] Overcomplete ICA trough convex optimisation


## References:

**Podosinnikova, A.**, Perry, A., Wein, A. S., Bach, F., d’Aspremont, A., & Sontag, D. (2019, April). Overcomplete independent component analysis via SDP. In The 22nd international conference on artificial intelligence and statistics (pp. 2583-2592). PMLR.

**Ding, C.**, Gong, M., Zhang, K., & Tao, D. (2019). Likelihood-free overcomplete ICA and applications in causal discovery. Advances in neural information processing systems, 32.
