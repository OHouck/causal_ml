# Intro to Torch/AD
# Can we automate derivatives?
# CML 2023
# Max Farrell & Sanjog Misra

# clear workspace
rm(list=ls())

library(torch)

# Infintesimal Jacknife (eps is a small perturbation)
# orginally from IF_Intuition.R
ij = function(data, stat, eps = 0.001) 
{
  n <- nrow(data)
  eps <- eps/n
  w.orig = rep(1,n)
  tobs <- stat(data, w.orig)
  L = NULL
  for (i in seq_len(n)) {
    w = w.orig

    # add small epsilon to one of the weights instead of setting it to 0
    w[i] <-  w[i]+eps
    L[i] <- n*(stat(data, w) - tobs)/eps
    }
  L
}

wmean = function(D,w){
  sum(D$x*w) / sum(w)
}

# Tensors and AD (automatic differentiation)
z <- torch_tensor(.5, requires_grad = TRUE)
y = z^2+z

y$backward()
# spits out the derivative of y with respect to z
z$grad

# Torch IF
set.seed(1234)
n <- 100
x0 <- matrix(runif(n),1,n)
x <- torch_tensor(x0)
w0 = matrix(1,n,1)
w = torch_tensor(w0, requires_grad=TRUE)

# mu = x'w / sum(w)
mu = torch_matmul(x,w)/torch_sum(w)

# differentiate mu with respect to w
grad <- autograd_grad(mu,w)
Lij = ij(data.frame(x=c(x0)),wmean)
Lauto = n*as.numeric(grad[[1]])

# plot shows we get the same answer
plot(Lij~Lauto,pch=19)
abline(0,1)




# Tensors and AD
z <- torch_tensor(.5, requires_grad = TRUE)
y = z^2+z
y$backward()
z$grad