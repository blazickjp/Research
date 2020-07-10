library(tidyverse)
library(patchwork)
library(invgamma)
library(gtools)
library(gridExtra)

p <- 0.4
mu_1 <- 9
mu_2 <- 15
sigma_1 <- 1
sigma_2 <- 2
n = 200


f <- function(mu_1, mu_2, sigma_1, sigma_2, p, n) {
  groups <- rbinom(n = n, size = 1, prob = p)
  
  df <- data.frame(group = groups) %>%
    rowwise() %>%
    mutate(y = if_else(group == 0, 
                       rnorm(n = 1, mean = mu_1, sd = sqrt(sigma_1)), 
                       rnorm(n = 1, mean = mu_2, sd = sqrt(sigma_2))))
  
  return(df$y)
  
}

df <- data.frame(y = f(mu_1, mu_2, sigma_1, sigma_2, p, n))

df %>%
  ggplot(aes(x = y)) + geom_histogram(aes(y = stat(density))) + 
  ggtitle("Mixture of 2 Gaussians") + 
  xlab("Observed Values") + ylab("Count") + 
  stat_function(fun = dnorm, args = list(mean = 9, sd = 1), col="red") + 
  stat_function(fun = dnorm, args = list(mean = 15, sd = 2), col="blue")


# Solve Mixture of 2 gaussians with EM algorithm
em_simple <- function(data, iters) {
  
  pi_i <- .7
  mu_1 <- 10
  mu_2 <- 11
  sigma_1 <- 5
  sigma_2 <- 5
  
  cat("pi", pi_i, "\nmu_1", mu_1, "\nsigma_1", sigma_1, "\nmu_2", mu_2, "\nsigma_2", sigma_2)
  
  for (k in 1:iters) {
    
    # E - Step
    p_a <- vector("numeric", length = nrow(data))
    p_b <- vector("numeric", length = nrow(data))
    
    for(i in 1:nrow(data)) { 
      y_a <- dnorm(data$y[i], mu_1, sqrt(sigma_1))
      y_b <- dnorm(data$y[i], mu_2, sqrt(sigma_2))
      
      p_a[i] <- pi_i*y_a / (pi_i*y_a + (1 - pi_i)*y_b)
      p_b[i] <- 1 - p_a[i]
    }
    
    # M - Step
    pi_i <- mean(p_a)
    
    sigma_1 <- sum(p_a * ((data$y - mu_1)^2)) / sum(p_a)
    sigma_2 <- sum(p_b * ((data$y - mu_2)^2)) / sum(p_b)
    
    mu_1 <- sum(p_a * data$y) / sum(p_a)
    mu_2 <- sum(p_b * data$y) / sum(p_b)
    
  }
  
  cat("\n\npi", pi_i, "\nmu_1", mu_1, "\nsigma_1", sigma_1, "\nmu_2", mu_2, "\nsigma_2", sigma_2)
  
  data %>%
    ggplot(aes(x = y)) + geom_histogram(aes(y = stat(density))) + 
    ggtitle("Mixture of 2 Gaussians", subtitle = glue::glue("Mu_1 = {mu_1} | Sigma_1 = {sigma_1}
                                                            mu_2 = {mu_2} | Sigma_2 = {sigma_2}
                                                            pi = {pi_i}")) + 
    xlab("Observed Values") + ylab("Count") + 
    stat_function(fun = dnorm, args = list(mean = mu_1, sd = sigma_1), col="red") + 
    stat_function(fun = dnorm, args = list(mean = mu_2, sd = sigma_2), col="blue") +
    stat_function(fun = dnorm, args = list(mean = 9, sd = 1), col = "black") + 
    stat_function(fun = dnorm, args = list(mean = 15, sd = sqrt(2)), col = "black")
  
}

em_simple(df, 20)

# Plot the Liklihood over iterations
# Metropolis Hastings if finishing the Gibbs quickly



gibbs <- function(data, iters) {
  
  pi_i <- rbeta(1,1,1)
  mu_1 <- rnorm(1,0,1)
  mu_2 <- rnorm(1,0,1)
  sigma_1 <- rinvgamma(1,1,1)
  sigma_2 <- rinvgamma(1,1,1)
  
  cat("pi", pi_i, "\nmu_1", mu_1, "\nsigma_1", sigma_1, "\nmu_2", mu_2, "\nsigma_2", sigma_2)
  
  sigma_1_vec <- vector("numeric", length = nrow(data))
  sigma_2_vec <- vector("numeric", length = nrow(data))
  mu_1_vec <- vector("numeric", length = nrow(data))
  mu_2_vec <- vector("numeric", length = nrow(data))
  pi_vec <- vector("numeric", length = nrow(data))
  
  for (k in 1:iters) {
    
    p_a <- vector("numeric", length = nrow(data))
    z <- vector("numeric", length = nrow(data))
    
    for(i in 1:nrow(data)) { 
      y_a <- dnorm(data$y[i], mu_1, sqrt(sigma_1))
      y_b <- dnorm(data$y[i], mu_2, sqrt(sigma_2))
      
      p_a[i] <- pi_i*y_a / (pi_i*y_a + (1 - pi_i)*y_b)
      z[i] <- sample(c(0,1), 1, replace = TRUE, prob = c(p_a[i], 1 - p_a[i]))
    }
    
    # Force at least one point into each group to avoid NA errors
    if (sum(z==0) == 0) { 
      z[sample(1:nrow(data), size = 1)] <- 0
    }
    if (sum(z==1) == 0){
      z[sample(1:nrow(data), size = 1)] <- 1
    }
    # Take samples from z | mu_1. mu_2, sigma_1, sigma_2
    mu_1 <- rnorm(1, mean = mean(data$y[z==0]), sd = sqrt(sigma_1)/sum(z==0))
    mu_2 <- rnorm(1, mean = mean(data$y[z==1]), sd = sqrt(sigma_2)/sum(z==1))
    mu_1_vec[k] <- mu_1
    mu_2_vec[k] <- mu_2
    
    # sigma_1 <- rchisq(1, sum(z==0)) / (sum(z==0) - 1)
    # sigma_2 <- rchisq(1, sum(z==1)) / (sum(z==1) - 1)
    sigma_1 <- rinvgamma(1, (sum(z==0) / 2), (.5*sum((data$y[z==0] - mu_1)^2)))
    sigma_2 <-  rinvgamma(1, (sum(z==1) / 2), (.5*sum((data$y[z==1] - mu_2)^2)))
    sigma_1_vec[k] <- sigma_1
    sigma_2_vec[k] <- sigma_2
    
    pi_i <- rbeta(1, 1 + mean(z==0), 1 + mean(z==1))
    pi_vec[k] <- pi_i
    
  }
  
  cat("\n\npi", mean(pi_i), "\nmu_1", mean(mu_1_vec), "\nsigma_1", mean(sigma_1_vec), 
      "\nmu_2", mean(mu_2_vec), "\nsigma_2", mean(sigma_2_vec))
  
  p_1 <- data %>%
    ggplot(aes(x = y)) + geom_histogram(aes(y = stat(density))) + 
    ggtitle("Mixture of 2 Gaussians", subtitle = glue::glue("Mu_1 = {mean(mu_1_vec)} | Sigma_1 = {mean(sigma_1_vec)}
                                                            mu_2 = {mean(mu_2_vec)} | Sigma_2 = {mean(sigma_2_vec)}
                                                            pi = {mean(pi_vec)}")) + 
    xlab("Observed Values") + ylab("Count") + 
    stat_function(fun = dnorm, args = list(mean = mean(mu_1_vec), sd = mean(sigma_1_vec)), col="red") + 
    stat_function(fun = dnorm, args = list(mean = mean(mu_2_vec), sd = mean(sigma_2_vec)), col="blue") +
    stat_function(fun = dnorm, args = list(mean = 9, sd = 1), col = "black") + 
    stat_function(fun = dnorm, args = list(mean = 15, sd = sqrt(2)), col = "black")
  
  p_2 <- data.frame(mu_1 = mu_1_vec,
                    mu_2 = mu_2_vec,
                    sigma_1 = sigma_1_vec,
                    sigma_2 = sigma_2_vec,
                    k = 1:iters) %>%
    gather(key, value, -k) %>%
    ggplot(aes(x = k, y = value, col = key, group = key)) + 
    geom_line() + 
    ggtitle("Parameter Samples by Iteration") + 
    xlab("Iteration") + ylab("Value") + 
    geom_hline(yintercept = 1) +
    geom_hline(yintercept = 2) + 
    geom_hline(yintercept = 9) + 
    geom_hline(yintercept = 15)
  
  p_1 / p_2
  
}


gibbs(df, 5000)

mh <- function(data, iters, n_warmup = 25) {
  
  # Set initial using Prior Distributions
  pi_i <- .4
  mu_1 <- rnorm(1,0,1)
  mu_2 <- rnorm(1,0,1)
  sigma_1 <- 1
  sigma_2 <- 2
  
  # Set vectors to keep track of parameter values over iterations 
  mu_1_vec <- vector("numeric", length = nrow(data))
  mu_2_vec <- vector("numeric", length = nrow(data))
  reject <- vector("numeric", length = iters)
  
  for (k in 1:iters) {
    # Set proposed using Prior Distributions
    mu_1_p <- mu_1 + rnorm(1, mean = 0, sd = .25)
    mu_2_p <- mu_2 + rnorm(1, mean = 0, sd = .25)
    
    log_loss_i <- vector("numeric", length = nrow(data))
    log_loss_p <- vector("numeric", length = nrow(data))
    p_a <- vector("numeric", length = nrow(data))
    p_b <- vector("numeric", length = nrow(data))
    
    for(i in 1:nrow(data)) { 
      # Loss per data point for initial
      y_a <- dnorm(data$y[i], mu_1, sqrt(sigma_1)) + 1e-5
      y_b <- dnorm(data$y[i], mu_2, sqrt(sigma_2)) + 1e-5
      
      p_a[i] <- pi_i*y_a / (pi_i*y_a + (1 - pi_i)*y_b)
      p_b[i] <- 1 - p_a[i]
      
      # Loss per data point
      log_loss_i[i] <- log(.4*dnorm(data$y[i], mu_1, sqrt(sigma_1)) + 
                             .6*dnorm(data$y[i], mu_2, sqrt(sigma_2)))
      
      y_a <- dnorm(data$y[i], mu_1_p, 1) + 1e-5
      y_b <- dnorm(data$y[i], mu_2_p, sqrt(2)) + 1e-5
      
      p_a[i] <- pi_i*y_a / (pi_i*y_a + (1 - pi_i)*y_b)
      p_b[i] <- 1 - p_a[i]
      
      # Loss per datapoint for proposed
      log_loss_p[i] <- log(.4*dnorm(data$y[i], mu_1_p, sigma_1) + 
                             .6*dnorm(data$y[i], mu_2_p, sqrt(sigma_2)))
    }
    if(is.na(sum(log_loss_p))) browser()
    alpha <- sum(log_loss_p) - sum(log_loss_i)
    c <- runif(1,0,1)

    if (log(c) < alpha) {
      reject[k] <- 0
      # Record values
      mu_1_vec[k] <- mu_1_p
      mu_2_vec[k] <- mu_2_p

      # Keep proposed parameters
      mu_1 <- mu_1_p
      mu_2 <- mu_2_p

    } else {
      # Record values
      reject[k] <- 1
      mu_1_vec[k] <- mu_1
      mu_2_vec[k] <- mu_2

    }
    
  }
  
  cat(
      "\nmu_1", mean(mu_1_vec[n_warmup:iters]), 
      "\nmu_2", mean(mu_2_vec[n_warmup:iters]), 
      "\nreject", mean(reject))
  # We throw out the first 25 values as "warm up" to prevent our estimates from being
  # Skewed by the variation at the begining
  p_1 <- data %>%
    ggplot(aes(x = y)) + geom_histogram(aes(y = stat(density)),binwidth = .5) + 
    ggtitle("Mixture of 2 Gaussians", 
            subtitle = glue::glue("mu_1 = {mean(mu_1_vec[n_warmup:iters])} | Sigma_1 = 1}
                                  mu_2 = {mean(mu_2_vec[n_warmup:iters])} | Sigma_2 = 2}
                                  pi = 0.4")) + 
    xlab("Observed Values") + ylab("Count") + 
    stat_function(fun = dnorm, args = list(mean = mean(mu_1_vec[n_warmup:iters]), 
                                           sd = 1), 
                  col="red") + 
    stat_function(fun = dnorm, args = list(mean = mean(mu_2_vec[n_warmup:iters]), 
                                           sd = sqrt(2)),
                  col="blue") +
    stat_function(fun = dnorm, args = list(mean = 9, 
                                           sd = 1), 
                  col = "black") + 
    stat_function(fun = dnorm, args = list(mean = 15, 
                                           sd = sqrt(2)), 
                  col = "black")
  
  p_2 <- data.frame(mu_1 = mu_1_vec,
                    mu_2 = mu_2_vec,
                    k = 1:iters) %>%
    gather(key, value, -k) %>%
    ggplot(aes(x = k, y = value, col = key, group = key)) + 
    geom_line() + 
    ggtitle("Parameter Samples by Iteration") + 
    xlab("Iteration") + ylab("Value") 
  
  grid.arrange(p_1, p_2, nrow = 2)
  browser()
}
mh(data = df, iters = 1000, n_warmup = 500)
 h