## OverICA

the R package `OverICA` estiamtes overcompelte indepedent components, meanign it estimates more altent variables then there are observed variables in the data. We observe p variabes in matrix y, we estimate the matrix A that contains the effects of k (non-gaussian) latent variables in x on the observed variables. there are potentially more variables k then p. 

y = Ax

this code estimates A by levraging the differences betwene the observed generalized covariance matrices and means in the data and the model implied generalized covariance matrices and means. This is based on ideas developed by Podosinnikova et al. (2019). 


Unlike Podosinnikova et al. I use back propagation to estimate the parameters. Based on ideas in Ging et al (2019) I define a generative neura network for each of the k latent variables (a multi=layer perceptron), and a matrix A that mixed these variables into p observed pseudo variables. I train the model to approzimate the the generalized covariacne matrices of the obersed data. Unlike Deng et al. I explicitly penalize the loss to ensure the latent variables remain uncorrelated.

## Cumulant Generating Function (CGF)

The **Cumulant Generating Function (CGF)** is a function that encodes all the cumulants of a random variable or a random vector. It is analogous to the moment generating function (MGF) but specifically tailored to generate cumulants.

### Definition

For a random vector \( \mathbf{X} = (X_1, X_2, \ldots, X_p)^T \), the CGF is defined as:

\[
K_{\mathbf{X}}(\mathbf{t}) = \log \mathbb{E}\left[ e^{\mathbf{t}^T \mathbf{X}} \right]
\]

where:
- \( \mathbf{t} = (t_1, t_2, \ldots, t_p)^T \) is a vector of real numbers.
- \( \mathbb{E} \) denotes the expectation operator.

### Properties

1. **Cumulants via Derivatives**: The cumulants of \( \mathbf{X} \) can be obtained by taking partial derivatives of the CGF evaluated at \( \mathbf{t} = \mathbf{0} \).

2. **Additivity for Independence**: If \( \mathbf{X} \) and \( \mathbf{Y} \) are independent random vectors, then:
   
   \[
   K_{\mathbf{X} + \mathbf{Y}}(\mathbf{t}) = K_{\mathbf{X}}(\mathbf{t}) + K_{\mathbf{Y}}(\mathbf{t})
   \]

3. **Relation to Moments**: While MGFs generate moments, CGFs generate cumulants, which are particularly useful for understanding dependencies and higher-order interactions.

## Covariance Matrix as Second-Order Cumulants

The **covariance matrix** is a fundamental measure of the pairwise linear relationships between the components of a random vector. It is directly related to the second-order cumulants of the distribution.

### Definition

For a random vector \( \mathbf{X} = (X_1, X_2, \ldots, X_p)^T \), the **covariance matrix** \( \Sigma \) is defined as:

\[
\Sigma = \mathbb{E}\left[ (\mathbf{X} - \mathbb{E}[\mathbf{X}]) (\mathbf{X} - \mathbb{E}[\mathbf{X}])^T \right]
\]

### Relation to Cumulants

The covariance matrix \( \Sigma \) comprises the second-order cumulants of the distribution of \( \mathbf{X} \). Specifically, the element \( \Sigma_{ij} \) is the covariance between \( X_i \) and \( X_j \), which is the second cumulant involving these two variables.

Thus, the covariance matrix can be viewed as capturing all second-order interactions (cumulants) within the random vector \( \mathbf{X} \).

## Mathematical Derivation

Let's delve into how the covariance matrix emerges from the CGF through differentiation.

### Step 1: CGF Definition

\[
K_{\mathbf{X}}(\mathbf{t}) = \log \mathbb{E}\left[ e^{\mathbf{t}^T \mathbf{X}} \right]
\]

### Step 2: First-Order Derivatives (Means)

Taking the first partial derivatives of the CGF with respect to \( t_i \) and \( t_j \), and evaluating at \( \mathbf{t} = \mathbf{0} \):

\[
\left. \frac{\partial K_{\mathbf{X}}(\mathbf{t})}{\partial t_i} \right|_{\mathbf{t}=0} = \mathbb{E}[X_i]
\]

This shows that the first cumulants are the means of the variables.

### Step 3: Second-Order Derivatives (Covariances)

Taking the second partial derivatives:

\[
\left. \frac{\partial^2 K_{\mathbf{X}}(\mathbf{t})}{\partial t_i \partial t_j} \right|_{\mathbf{t}=0} = \text{Cov}(X_i, X_j)
\]

#### Derivation:

1. **First Derivative**:

   \[
   \frac{\partial K_{\mathbf{X}}(\mathbf{t})}{\partial t_i} = \frac{1}{\mathbb{E}[e^{\mathbf{t}^T \mathbf{X}}]} \cdot \mathbb{E}\left[ X_i e^{\mathbf{t}^T \mathbf{X}} \right]
   \]

2. **Second Derivative**:

   \[
   \frac{\partial^2 K_{\mathbf{X}}(\mathbf{t})}{\partial t_i \partial t_j} = \frac{\mathbb{E}\left[ X_i X_j e^{\mathbf{t}^T \mathbf{X}} \right]}{\mathbb{E}\left[ e^{\mathbf{t}^T \mathbf{X}} \right]} - \frac{\mathbb{E}\left[ X_i e^{\mathbf{t}^T \mathbf{X}} \right] \mathbb{E}\left[ X_j e^{\mathbf{t}^T \mathbf{X}} \right]}{\left( \mathbb{E}\left[ e^{\mathbf{t}^T \mathbf{X}} \right] \right)^2}
   \]

   Evaluating at \( \mathbf{t} = \mathbf{0} \):

   \[
   \left. \frac{\partial^2 K_{\mathbf{X}}(\mathbf{t})}{\partial t_i \partial t_j} \right|_{\mathbf{t}=0} = \mathbb{E}[X_i X_j] - \mathbb{E}[X_i] \mathbb{E}[X_j] = \text{Cov}(X_i, X_j)
   \]




References:

Podosinnikova, A., Perry, A., Wein, A. S., Bach, F., d’Aspremont, A., & Sontag, D. (2019, April). Overcomplete independent component analysis via SDP. In The 22nd international conference on artificial intelligence and statistics (pp. 2583-2592). PMLR.

Ding, C., Gong, M., Zhang, K., & Tao, D. (2019). Likelihood-free overcomplete ICA and applications in causal discovery. Advances in neural information processing systems, 32.
