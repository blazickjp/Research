---
layout: post
title:  "Infinite Component Mixtures and The Dirirchlet Process"
categories: mypost
---

## Extending to Infinite Components

After working with mixture models, it's natural to wonder how one should best determine the correct number of components to include in our models. One method for determining this is to extend our finite mixture model to the case of infinite components. Thinking back to the finite case we wonder how this could be possible because we are defining $\pmb{\pi}$ as a finite probability vector from which we sample $z_i$. This is where the Dirichlet Process comes in.

## Dirichlet Processes

A Dirichlet Process is best describred as a stochastic process used in bayesian non-parametric modeling, and more specifically in Dirichlet Process Mixture Models (Infinite Mixture Models). Each draw from a DP is itselt a distribution, making it a distribution over distributions. The DP derives it's name from the fact that the marginal distributions of a DP is a finite dimensional Dirichlet Distribution just as a Gaussian Process has a finite dimensional Gaussian distributed marginal distribution. The DP has an infinite number of parameters which places it in the family of non-parametrics.

## Stick Breaking Definition

One common way to define the Dirichlet Process is by using the popular "stick breaking" construction. Formally, Let $(\phi_1,\phi_2, ...)$ be a sequence of independent random variables distributed $Beta(1, \alpha)$. Independent of this sequence 
let $(Z_1, Z_2, ...)$ be a sequence of random variables with base distribution $H$. If we define $p1 = \phi_1$ and $p_i = \phi_i\prod_{j\lti}(1-\phi_j)$.