# Author: Ozzy Houck
# Date: 12/7/2023

# Purpose: This is the code submitted as a part of the causal machin learning 
# final project. It's purpose is to create a 95% confidence interval for the 
# marginal treatment effect of a treatment variable in a logistic model

# Code is build off of the in class code from week 8. 

# clear workspace
rm(list=ls())

# code location
code  <- "~/vc/causal_ml/"

# load libraries
library(torch)
library(MASS)

# Set seed
set.seed(343532)
torch_manual_seed(235324)

#==============================================================================
# 1. Generate data
#==============================================================================

# input dimensionality (number of input features 'X')
d_in <- 5

# treatment dimensionality (number of input features 'Z')
d_z <- 1L

# output dimensionality (dim of [alpha,beta] : for now 2)
d_out  <- 2 

# number of observations in training set
N <- 25000 

# create random data
x  <- torch_tensor(matrix(rnorm(N*d_in), N, d_in))

# Generate alpha(x) using some arbitrary function
true_a <- -1 + x[, 1, NULL] - 0.5 * torch_pow(x[, 2, NULL], 2)

# Generate beta(x) also arbitrarily
true_b <- 2 + x[, 3, NULL] + 2 * x[, 4, NULL]

# true_a <- -0.5 - x[, 1, NULL] + torch_sigmoid(x[, 2, NULL])
# true_a <- 3 - 0.2 * x[, 3, NULL] + torch_pow(x[, 4, NULL], 2)

# Assign treatment t(X) = <placeholder>
# t needs to be strictly positive and continuous
# z <- torch_pow(torch_randn(N, 1L) + x[, 5, NULL], 2) + 0.5

# uniform between .2 and .8
z <- torch_rand(N, 1L) * .6 + 0.2


#---------------------------------------------
# linear model for testing
# y_prob <- function(alpha, beta, z) {
#   alpha + torch_mul(beta, z) %>% return()
# }
# true_prob <- y_prob(true_a, true_b, z) + torch_randn(N, 1)

# y <- torch_tensor( # recast as torch tensor
#   as.matrix(true_prob),
#   dtype = torch_float()
# )

# loss_fn <- function(y_pred, y, reduction = "mean") {
#   nnf_mse_loss(y_pred, y, reduction = reduction) %>% return()
# }
#---------------------------------------------


# function to evaltuate probability of y
y_prob <- function(alpha, beta, z) {
    # y_prob = 1/(1+exp(-alpha-beta*ln(z)))
    y_prob  <-  torch_reciprocal(
        torch_add(1, torch_exp(
                            torch_subtract(torch_mul(-1, alpha), 
                            torch_mul(beta, torch_log(z)))
                            )
                    )
    )  %>% return()
}
# calculate the true probability of y
true_prob <- y_prob(true_a, true_b, z)

y  <- rbinom(N, 1, prob = as.numeric(true_prob))
y  <- torch_tensor(as.matrix(y)) 

# loss function is negative log likelihood
loss_func <- function(y_pred, y, reduction) {
    # loss = -y * log(y_pred) - (1 - y) * log(1 - y_pred)

    # check if y_pred is 0 or 1
    # if so, add a small number to avoid log(0)
    y_pred <- torch_where(y_pred == 0, torch_tensor(0.01), y_pred)
    y_pred <- torch_where(y_pred == 1, torch_tensor(0.99), y_pred)

    # loss  <- -y * torch_log(y_pred) - (1 - y) * torch_log(1 - (y_pred))
#   loss <- -y * torch_log(y_pred + 0.0001) - (1 - y) *
#     torch_log(1 + 0.0001 - y_pred)

    loss <- torch_sub(torch_mul(torch_neg(y), torch_log(y_pred)), torch_mul(torch_sub(1, y), torch_log(torch_sub(1, y_pred))))


  if (reduction == "mean") {
    # loss <- loss * (1 / (y %>% length()))
    loss  <- torch_mean(loss) %>% return()
  }

  # other option is sum
  else {
    torch_sum(loss) %>% return()
  }
}

#==============================================================================
# 2. Create sample splits
#==============================================================================

full_dat  <- list()
full_dat$x  <- x
full_dat$y  <- y
full_dat$z  <- z

# sample split into 3 splits
idxc= idx=list()
idx[[1]] = sample(1:N,size = N/3,replace=FALSE)
idxc[[1]] = setdiff(1:N,idx[[1]])

idx[[2]] = sample(idxc[[1]],size = N/3,replace=FALSE)
idxc[[2]] = setdiff(1:N,idx[[2]])

idx[[3]] = setdiff(idxc[[1]],idx[[2]])
idxc[[3]] = setdiff(1:N,idx[[3]])

# Create splits
split_1 <- list()
split_1$x <- x[idx[[1]], ]
split_1$y <- y[idx[[1]], ]
split_1$z <- z[idx[[1]], ]

split_2 <- list()
split_2$x <- x[idx[[2]], ]
split_2$y <- y[idx[[2]], ]
split_2$z <- z[idx[[2]], ]

split_3 <- list()
split_3$x <- x[idx[[3]], ]
split_3$y <- y[idx[[3]], ]
split_3$z <- z[idx[[3]], ]

#==============================================================================
# 3. Load functions to implement causalDNN
#==============================================================================

source(paste0(code, "houck_final/cdnn_houck.R")) # Step 1
source(paste0(code, "session_8/linDNN.R")) # Step 2
source(paste0(code, "houck_final/projLam_houck.R")) 
source(paste0(code, "houck_final/procRes_houck.R")) # Step 3

# Crossfits
# 1, 2, 3
# 3, 1 ,2
# 2, 3, 1

# Run DeepNets
dnn1  <- cdnn(split_1, y_pred_func = y_prob, loss_func = loss_func,
              learn_rate = 0.01, weight_decay = 1e-3, 
              arch = c(20, 20, 20, 20, 20, 20, 20, 20, 20, 20))

dnn2  <- cdnn(split_2, y_pred_func = y_prob, loss_func = loss_func,
              learn_rate = 0.01, weight_decay = 1e-3,
              arch = c(20, 20, 20, 20, 20, 20, 20, 20, 20, 20))

dnn3  <- cdnn(split_3, y_pred_func = y_prob, loss_func = loss_func,
              learn_rate = 0.01, weight_decay = 1e-3,
              arch = c(20, 20, 20, 20, 20, 20, 20, 20, 20, 20))

# Get parameters on full data
aba  <- dnn1$model(x)
abb  <- dnn2$model(x)
abc  <- dnn3$model(x)

#==============================================================================
# 4. Compute influence functions and do inference
#==============================================================================

# Projections
# Projects the Hessian onto X
# To form conditional expectation functions
# make minor changes to projLam.R to take in differnet 
# loss functions and prediction functions
lProj1 = makeLam(dat = split_2,dnn=dnn3, 
    loss_func = loss_func, y_pred_func = y_prob,
    arch = c(20, 20, 20, 20, 20, 20, 20, 20, 20, 20))
lProj2 = makeLam(dat = split_2,dnn=dnn1, 
    loss_func = loss_func, y_pred_func = y_prob,
    arch = c(20, 20, 20, 20, 20, 20, 20, 20, 20, 20))
lProj3 = makeLam(dat = split_2,dnn=dnn2, 
    loss_func = loss_func, y_pred_func = y_prob,
    arch = c(20, 20, 20, 20, 20, 20, 20, 20, 20, 20))

# Compute IF for each split and stack
# What statistic are we interested in
# Let's say ATE = (E[H]=E(CATE)=E(ab[,2]))
# H=function(ab, full_dat) ab[,2]

# H for marginal effect
H  <-  function(theta, data) {
    # H = (e^a * b * t^(b-1)) / (e^a * t^b + 1)^2
    n = nrow(data$x)
    alpha <- theta[,1]$reshape(c(n,1L))
    beta <- theta[,2]$reshape(c(n,1L))

    H  <- torch_div(
        torch_mul(
            torch_mul(
                torch_exp(alpha), beta
            ), 
            torch_pow(data$z, torch_subtract(beta, 1))
        ), 
        torch_pow(
            torch_add(
                torch_mul(
                    torch_exp(alpha), 
                    torch_pow(data$z, beta)
                ), 
                1
            ), 
            2
        )
    ) %>% return()
}

# Use split 3s for 
fin3 = proc_res(split_3 ,dnn1,lProj2,H, loss_func = loss_func, y_pred_func = y_prob)
fin2 = proc_res(split_2,dnn3,lProj1,H, loss_func = loss_func, y_pred_func = y_prob)
fin1 = proc_res(split_1,dnn2,lProj3,H, loss_func = loss_func, y_pred_func = y_prob)

# Stack 
af1 = fin1$auto.if
af2 = fin2$auto.if
af3 = fin3$auto.if
cf.est= mean(c(af1,af2,af3))
cf.se = sqrt((1/3)*(var(af1)+var(af2)+var(af3))/N)

cf = c(Est=cf.est,se = cf.se,
       CI.L=cf.est-1.96*cf.se,CI.U=cf.est+1.96*cf.se)

print("CF")
print(cf)

#Truth
    # H = (e^a * b * t^(b-a)) / (e^a * t^b + 1)^2

b  <- as.numeric(true_b)
a  <- as.numeric(true_a)
z  <- as.numeric(z)
H = (exp(a) * b * z^(b-1)) / (exp(a) * z^b + 1)^2

print(paste0("Truth: ", mean(H)))

# save truth and estimate in a text file
write.table(c(paste0("estimate: ",cf.est), 
              paste0("se: ", cf.se),
              paste0("truth: ", mean(H))
              ), file = paste0(code, "houck_final/cf.txt"), 
              row.names = FALSE, col.names = FALSE)
