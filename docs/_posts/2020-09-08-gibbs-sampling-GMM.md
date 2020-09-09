---
layout: post
title:  "Deriving Complete Conditionals for GMM Gibbs Sampler"
categories: mypost
---

## Gibbs Sampling

Gibbs sampling is a Markov Chain Monte Carlo method for sampling from a posterior distribution usually defined 
as $p(\theta|data)$. The idea behind the Gibbs Sampler is to sweep through each one of the parameters and sample from their conditional distributions, fixing the other parameters constant. For example, consider the random variables $X_1, X_2, X_3$ and assume that I can write out the analytic form of $p(X_1|X_2,X_3), p(X_2|X_1,X_3), p(X_3|X_2,X_1)$ . We start by initializing $x_{1,t}, x_{2,t}, x_{3,t}$ and for each iteration $t$ we sample $p(X_{1,t+1}|X_{2,t},X_{3,t})$, $p(X_{2,t+1}|X_{1,t+1},X_{3,t})$, and $p(X_{3,t+1}|X_{2,t+1},X_{3,t+1})$. This process can then continue until convergence.

## Mixture of Normals

Now that we understand the ideas behind Gibbs Sampling, let's determine how we can use it to fit a mixture of 2 univariate gaussians. Our model is defined by $p(x|\theta) = \pi\phi_{\theta_i}(x) + (1-\pi)\phi_{\theta_2}(x)$. This just means that we have some probability $\pi$ of taking our observation from $\phi_{\theta_1}(x)$ where $$\phi_{\theta_1}(x) \sim N(\mu_1, \sigma^2_1)$$ and $(1-\pi)$ probability of coming 
from $\phi_{\theta_2}(x)$ where $\phi_{\theta_2}(x) \sim N(\mu_2, \sigma^2_2)$. Using python we can show this as follows:

```python
import numpy as np
from numpy.random import binomial, normal, beta, multinomial
import scipy.stats as st
from scipy.stats import invgamma, norm
import matplotlib.pyplot as plt
from distcan import InverseGamma

def data_gen(mu, sigmas, phi, n):
    """
    Generates samples from Mixture of 2 Gaussian Distributions
    """
    y = []
    for i in range(n):
        ind = binomial(1, phi, 1)
        if ind == 1:
            y.append(norm(mu[0], sigmas[0]).rvs())
        else:
            y.append(norm(mu[1], sigmas[1]).rvs())
    return np.array(y)

# Set Starting Parameters
mu = [0, 8]
sigmas = [1, 3]
phi = .4
n = 500
y = data_gen(mu=mu, sigmas=sigmas, phi=phi, n=n)
x = np.linspace(-3,14)

# Create Plot of Data 
plt.hist(y, 30, density=True, alpha=0.5)
plt.plot(x, norm(mu[0], sigmas[0]).pdf(x), color="red")
plt.plot(x, norm(mu[1], sigmas[1]).pdf(x), color="blue")
plt.title("Mixture of 2 Gaussians Data")
plt.grid()
plt.show()
```
![svg](images/output_5_0.svg)

It can be very difficult to calculate the posterior under conjugate priors for a normal mixture model, so instead we can use a ${0,1}$ indicator variable $Z$ to make the calculations easier. 

If we let $\theta_j = \{\mu_j,\sigma^2_j,\pi\}$ we see that the joint density: 

$$p(x, z; \theta) = p(x|z,\theta) p(z,\theta)$$

where: 
$$p(x|z,\theta) = \phi_{\theta_1}(x)^{z_1}\phi_{\theta_2}(x)^{z_2}$$. And $p(z,\theta)$ comes from the multinomial distribution with density $\frac{n!}{x_1!x_2!}\pi_1^{z_1}\pi^{z_2}$. Because $z$ is an indicator variable, $\frac{n!}{x_1!x_2!} = 1$ so our second term is given by:

$$
\begin{align*}
p(z,\theta) & = \pi^{z_1}(1-\pi)^{z_2}\\
p(z,\theta) & = \prod_{j=1}^K\pi^{z_j}
\end{align*}
$$

Which gives the joint density over $x,z$ as:
$$p(x, z; \theta) = \prod_{i=1}^N\left[\pi\phi_{\theta_1}(x_i)\right]^{z_1}\left[(1-\pi)\phi_{\theta_2}(x_i)\right]^{z_2}$$
We can now define our prior distributions using conjugacy. Using conjugate priors is helpful because it allows us to easily compute the posterior. We define our priors over $\{\mu_j,\sigma^2_j,\pi\}$ as follows:

$$
\begin{align*}
p(\pi) & \sim Beta(\alpha = 1, \beta = 1)\\
p(\mu_j) & \sim N(\mu_0 = 0, \tau^2 = 1)\\
p(\sigma_j^2) & \sim IG(\delta = 1, \psi = 1)
\end{align*}
$$

To make life easier, we can plug our priors directly into our densities to get the following densities over our priors:

$$
\begin{align*}
p(\pi|\alpha,\beta) & = \pi^{\alpha-1}(1-\pi)^{\beta-1}\\
& \propto const\\
p(\mu_j|\mu_0,\tau^2) & = \frac{1}{\sqrt{2\pi\tau^2}}\exp\left[-\frac{(\mu_j - \mu_0)^2}{2\tau^2}\right]\\
& \propto \exp\left[-\frac{\mu_j^2}{2\tau^2}\right]\\
p(\sigma_j^2|\delta, \psi) & = \left(\sigma^2_j\right)^{-\delta - 1}\exp\left[-\frac{\psi}{\sigma^2_j}\right]\\
& \propto \left(\sigma^2_j\right)^{-2}\exp\left[-\frac{1}{\sigma^2_j}\right]
\end{align*}
$$

Which leads to the posterior distribution of $\theta$ where $\theta = \{\pi, \mu, \sigma^2\}$:

$$
\begin{align*}
p(\theta|x,z) & = p(x, z| \theta)p(\pi)\prod_{j=1}^k\left[p(\mu_j)p(\sigma_j^2)\right]\\
& = \prod_{i=1}^N\pi^{z_1}\phi_{\theta_1}(x_i)^{z_1}\prod_{i=1}^N(1-\pi)^{z_1}\phi_{\theta_2}(x_i)^{z_2}\\
& = \prod_{i=1}^N\pi^{z_1}\prod_{i=1}^N\pi^{z_2}\prod_{i=1}^N\phi_{\theta_1}(x_i)^{z_1}\prod_{i=1}^N\phi_{\theta_2}(x_i)^{z_2}\exp\left[-\frac{\mu_j^2}{2\tau^2}\right]\left(\sigma^2_j\right)^{-2}\exp\left[-\frac{1}{\sigma^2_j}\right]\\
& = \pi^{\sum_{i=1}^Nz_1}(1-\pi)^{\sum_{i=1}^Nz_2}\prod_{i=1}^N\prod_{j=1}^K\phi_{\theta_j}(x_i)^{z_j}\prod_{j=1}^K\exp\left[-\frac{\mu_j^2}{2\tau^2}\right]\left(\sigma^2_j\right)^{-2}\exp\left[-\frac{1}{\sigma^2_j}\right]\\
\end{align*}
$$