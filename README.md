## OverICA

the R package `OverICA` estiamtes overcompelte indepedent components, meanign it estimates more altent variables then there are observed variables in the data. We observe p variabes in matrix y, we estimate the matrix A that contains the effects of k (non-gaussian) latent variables in x on the observed variables. there are potentially more variables k then p. 

y = Ax

this code estimates A by levraging the differences betwene the observed generalized covariance matrices and means in the data and the model implied generalized covariance matrices and means. This is based on ideas developed by Podosinnikova et al. (2019). 


Unlike Podosinnikova et al. I use back propagation to estimate the parameters. Based on ideas in Ging et al (2019) I define a generative neura network for each of the k latent variables (a multi=layer perceptron), and a matrix A that mixed these variables into p observed pseudo variables. I train the model to approzimate the the generalized covariacne matrices of the obersed data. Unlike Deng et al. I explicitly penalize the loss to ensure the latent variables remain uncorrelated.


# Understanding the Covariance Matrix via the Cumulant Generating Function

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
   
   $$
   \frac{\partial^2 K_{\mathbf{X}}(\mathbf{t})}{\partial t_i \partial t_j} \bigg|_{\mathbf{t}=0} = \text{Cov}(X_i, X_j) = \Sigma_{ij}
   $$
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



**References:**

References:

Podosinnikova, A., Perry, A., Wein, A. S., Bach, F., d’Aspremont, A., & Sontag, D. (2019, April). Overcomplete independent component analysis via SDP. In The 22nd international conference on artificial intelligence and statistics (pp. 2583-2592). PMLR.

Ding, C., Gong, M., Zhang, K., & Tao, D. (2019). Likelihood-free overcomplete ICA and applications in causal discovery. Advances in neural information processing systems, 32.
