---
layout: post
title:  "Extending K-Component GMM to the Bivariate Case"
categories: mypost
---

## Introduction

In my previous post, we extended our Gibbs Sampler to handle any number of $K$ Components and fit our model in the case where $K=4$. We can now move on and extend our model to handle the Multivariate case. I stick to demonstrating the model on the bivariate case for simplicity and to better enable  visualization of the results. In order to understand how this works, we'll need to introduce and understand the Inverse-Wishart distribution. 

## Inverse-Wishart Distribution

Need to put more here...


## Complete Conditional for $p(\pmb{\mu_j})$ and $p(\Sigma_j)$

We know from our last post that 
$$
\begin{align*}
p(\theta|x,z) \propto p(x, z| \theta)p(\pmb{\pi})\prod_{j=1}^k\left[p(\pmb{\mu_j})p(\Sigma_j)\right]
\end{align*}
$$ 

and we can define the joint prior density

$$
\begin{align*}
p(\mu_j, \Sigma_j) & \propto \left|\Sigma_j\right|^{-\frac{v_j + d}{2+1}}\exp\left(\frac{1}{2}tr(\Lambda_j\Sigma_j^{-1})- \frac{\kappa_j}{2}(\mu_j - \xi_j)^T\Sigma_j^{-1}(\mu_j - \xi_j)\right)
\end{align*}
$$

which leads to the conditional density

$$
\begin{align*}
p(\mu_j, \Sigma_j | z, x, \pi) & \propto \prod_{i=1}^N\phi_{\theta_j}(x_i)^{z_j}\left|\Sigma_j\right|^{-\frac{v_j + d}{2+1}}\exp\left(\frac{1}{2}tr(\Lambda_j\Sigma_j^{-1})- \frac{\kappa_j}{2}(\mu_j - \xi_j)^T\Sigma_j^{-1}(\mu_j - \xi_j)\right)
\end{align*}
$$

This results in simply multiplying our joint prior density with a multivariate normal. Because of conjugacy, we know this results in a posterior density of the same family (normal inverse-Wishart) with parameters:

$$
\begin{align*}
\mu_n & = \frac{\kappa_j}{\kappa_j + n_j}\xi_j + \frac{n_j}{\kappa_j + n_j}\bar{y}\\
\kappa_n & = \kappa_j + n_j\\
v_n & = v_j + n_j\\
\Lambda_n & = \Lambda_j + \sum_{i=1}^N(x_i - \bar{x})(x_i - \bar{x})^T + \frac{n_j\kappa_j}{\kappa_j + n_j}(\bar{x} - \xi_j)(\bar{x} - \xi_j)^T
\end{align*}
$$

Samples from the joint conditional distribution $p(\mu_j, \Sigma_j | z, x, \pi)$ can be obtained by first sampling 
from $p(\Sigma_j) \sim W^{-1}(v_n, \Lambda_n)$, then sampling $p(\pmb{\mu}_j) \sim N(\pmb{\mu}_n, \frac{\Sigma_j}{\kappa_n})$
