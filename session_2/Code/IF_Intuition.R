# Basic Inference Ideas
# Connecting Influence functions to Inference
# CML
# Max Farrell & Sanjog Misra
set.seed(1235)
N=100
n = N
df = data.frame(x = runif(N))

# Sample Mean
sample.mean = mean(df$x)
sample.sd = sd(df$x)
sample.se = sample.sd/sqrt(n)

# Let's construct a weighted mean function mu(D) = wmean(D)
# assigning weights to different rows using uniform weights
wmean = function(D,w){
  sum(D$x*w) / sum(w)
}

# So...
wmean(df,rep(1,N))
sample.mean

# Theory
L = df$x - mean(df$x)


# The (LOO) Jackknife
jack = function(data, stat) 
{
  n <- nrow(data)
  w.orig = rep(1,n)
  tobs <- stat(data, w.orig)
  L = NULL
  for (i in seq_len(n)) {
    w = w.orig
    w[i] <-  0
    L[i] <- n*tobs - (n-1)*stat(data, w)
  }
  L - tobs # De-mean
}

Lj = (jack(df,wmean))
mean(Lj)
sqrt(var(Lj)/n)

# Infintesimal Jacknife (eps is a small perturbation)
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

Lij = ij(df,wmean)

# infintisimial jacknife as larger mean (both very close to 0)
mean(Lj)
mean(Lij)

# same standard errors
sqrt(var(Lij)/n)
sqrt(var(Lj)/n)

# Using package boot
library(boot)

# This does the infintesimal jacknife
L0 = empinf(data = df, statistic = wmean, type = "inf", stype = "i")
# L0 = empinf(data = df, statistic = wmean)
plot(Lij~L0); abline(0,1)

# Estimates
est = mean(df$x+Lij)
se = sqrt(var(Lij)/N)
ci = c(est-1.96*se, est+1.96*se)



