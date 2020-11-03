---
layout: post
title:  "Extending K-Component GMM to the Bivariate Case"
categories: mypost
---

## Introduction

In my previous post, we extended our Gibbs Sampler to handle any number of $K$ Components and fit our model in the case where $K=4$. We can now move on and extend our model to handle the Multivariate case. I stick to demonstrating the model on the bivariate case for simplicity and to better enable  visualization of the results. In order to understand how this works, we'll need to introduce and understand the Inverse-Wishart distribution. 

## Inverse-Wishart Distribution

