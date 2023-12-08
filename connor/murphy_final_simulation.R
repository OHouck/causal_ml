# murphy_final_simulation.R
# Author: Connacher Murphy
# Simulation for Causal ML Final (Fall 2023)
# Based on methods of Farrell, Liang, and Misra (2021)
# Based on code from Farrell and Misra Causal ML course
# See https://arxiv.org/abs/2010.14694

# CM: add file headings

rm(list = ls())

library(torch) # required libraries
library(MASS)
library(ggplot2)

path <- "~/projects/causal_ml/connor"

#------------------------------------------------------------------------------#
# 1. Data generating process
#------------------------------------------------------------------------------#
loss_func <- function(y_pred, y, reduction) {
  # nnf_mse_loss(y_pred, y, reduction = reduction) %>% return()
  # Use binary cross entropy loss for the binary classification final problem
  nnf_binary_cross_entropy(y_pred, y, reduction = reduction) %>% return()
}

y_pred_func <- function(alpha, beta, z) { # generates prediction for y
  y_prob <- torch_reciprocal(
    torch_add(1, torch_mul(
      torch_exp(torch_neg(alpha)),
      torch_pow(z, torch_neg(beta))
    ))
  ) %>% return()
}

d_in <- 5 # input dimensionality: dim(x)
dim_z <- 1L # treatment dimensionality: dim(Z)
d_out <- 2 # output dimensionality: dim(theta)

N <- 60000  # number of observations in training set

set.seed(12345)
torch_manual_seed(12345)

x <- torch_tensor(matrix(rnorm(N * d_in), N, d_in)) # draw X from normal dist
true_alpha <- -0.5 - x[, 1, NULL] + torch_sigmoid(x[, 2, NULL])
true_beta <- 3 - 0.2 * x[, 3, NULL] + torch_pow(x[, 4, NULL], 2)

z <- torch_pow(torch_exp(0.8 * torch_randn(N, 1L)), 0.5) + 0.1 # treatment

true_prob <- y_pred_func(true_alpha, true_beta, z) # calculate probability

plt <- ggplot() + # density of P{Y = 1}
  geom_histogram(aes(x = true_prob %>% as.numeric()),
                 color = "deepskyblue3", fill = "deepskyblue") +
  labs(x = "true_prob", y = "Count", title = "Distribution of Pr{Y = 1}") +
  theme_bw()
ggsave(file.path(path, "out", "density_true_prob.png"), plt)

y <- rbinom(N, 1, prob = as.numeric(true_prob)) # draw from Bernoulli
y <- torch_tensor( # recast as torch tensor
  as.matrix(y),
  dtype = torch_float()
)

table(as.numeric(y)) # tabulate binary outcome y

#------------------------------------------------------------------------------#
# 2. Create sample splits
#------------------------------------------------------------------------------#
# data <- list() # full dataset (don't think we need this for present purpose)
# data$x <- x
# data$y <- y
# data$z <- z

idxc <- idx <- list() # Sample IDs for 3 splits
idx[[1]] <- sample(1:N, size = N / 3, replace = FALSE) # first split
idxc[[1]] <- setdiff(1:N, idx[[1]]) # complement of the first split

# Use complement of the first split to ensure no overlap
idx[[2]] <- sample(idxc[[1]], size = N / 3, replace = FALSE)
idxc[[2]] <- setdiff(1:N, idx[[2]])

idx[[3]] <- setdiff(idxc[[1]], idx[[2]])
idxc[[3]] <- setdiff(1:N, idx[[3]])

split_1 <- list() # create 3 splits
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

#------------------------------------------------------------------------------#
# 3. Causal DNN functions
#------------------------------------------------------------------------------#
source(file.path(path, "linear_DNN.R")) # linear DNN used in step 2

source(file.path(path, "theta_DNN.R")) # step 1
source(file.path(path, "est_Lambda.R")) # step 2
source(file.path(path, "influence_function.R")) # step 3

#------------------------------------------------------------------------------#
# 4. Implement causal DNN - thetaDNN
#------------------------------------------------------------------------------#
# Crossfits (CM: check crossfit sequences in all sections below):
# 1, 2, 3
# 3, 1, 2
# 2, 3, 1

# Future improvements:
# Switch Adam optimization parameters in function arguments
# Experiment more with different architectures
# Add in regularization

# Estimate theta with DNN
cat("Split 1:\n")
dnn_1 <- theta_DNN(split = split_1, loss_func = loss_func,
                   y_pred_func = y_pred_func, arch = c(30, 30, 30, 30, 30, 30),
                   N_epoch = 1000)
cat("Split 3:\n")
dnn_3 <- theta_DNN(split = split_3, loss_func = loss_func,
                   y_pred_func = y_pred_func, arch = c(30, 30, 30, 30, 30, 30),
                   N_epoch = 1000)
cat("Split 2:\n")
dnn_2 <- theta_DNN(split = split_2, loss_func = loss_func,
                   y_pred_func = y_pred_func, arch = c(30, 30, 30, 30, 30, 30),
                   N_epoch = 1000)

theta_1 <- dnn_1$model(x) # calculate parameters for full data
theta_2 <- dnn_2$model(x)
theta_3 <- dnn_3$model(x)

# Diagnostic plots
a_e_1 <- theta_1[, 1] %>% as.numeric() # estimated alpha
b_e_1 <- theta_1[, 2] %>% as.numeric() # estimated beta
a_e_2 <- theta_2[, 1] %>% as.numeric()
b_e_2 <- theta_2[, 2] %>% as.numeric()
a_e_3 <- theta_3[, 1] %>% as.numeric()
b_e_3 <- theta_3[, 2] %>% as.numeric()

a_t <- true_alpha %>% as.numeric() # true alpha
b_t <- true_beta %>% as.numeric() # true beta

plt <- ggplot() +
  geom_point(aes(x = a_e_1, y = a_t), size = 0.25) +
  geom_abline(intercept = 0, slope = 1) +
  labs(x = "Estimated alpha", y = "True alpha", title = "Split 1") +
  theme_bw()

ggsave(file.path(path, "out", "est_alpha_1.png"), plt)

plt <- ggplot() +
  geom_point(aes(x = a_e_2, y = a_t), size = 0.25) +
  geom_abline(intercept = 0, slope = 1) +
  labs(x = "Estimated alpha", y = "True alpha", title = "Split 1") +
  theme_bw()

ggsave(file.path(path, "out", "est_alpha_2.png"), plt)

plt <- ggplot() +
  geom_point(aes(x = a_e_3, y = a_t), size = 0.25) +
  geom_abline(intercept = 0, slope = 1) +
  labs(x = "Estimated alpha", y = "True alpha", title = "Split 1") +
  theme_bw()

ggsave(file.path(path, "out", "est_alpha_3.png"), plt)

plt <- ggplot() +
  geom_point(aes(x = b_e_1, y = b_t), size = 0.25) +
  geom_abline(intercept = 0, slope = 1) +
  labs(x = "Estimated beta", y = "True beta", title = "Split 1") +
  theme_bw()

ggsave(file.path(path, "out", "est_beta_1.png"), plt)

plt <- ggplot() +
  geom_point(aes(x = b_e_2, y = b_t), size = 0.25) +
  geom_abline(intercept = 0, slope = 1) +
  labs(x = "Estimated beta", y = "True beta", title = "Split 2") +
  theme_bw()

ggsave(file.path(path, "out", "est_beta_2.png"), plt)

plt <- ggplot() +
  geom_point(aes(x = b_e_3, y = b_t), size = 0.25) +
  geom_abline(intercept = 0, slope = 1) +
  labs(x = "Estimated beta", y = "True beta", title = "Split 3") +
  theme_bw()

ggsave(file.path(path, "out", "est_beta_3.png"), plt)

plt <- ggplot() +
  geom_density(aes(x = a_e_1, color = "Split 1"), show.legend = FALSE) +
  stat_density(aes(x = a_e_1, color = "Split 1"),
               geom = "line", position = "identity") +
  geom_density(aes(x = a_e_2, color = "Split 2"), show.legend = FALSE) +
  stat_density(aes(x = a_e_2, color = "Split 2"),
               geom = "line", position = "identity") +
  geom_density(aes(x = a_e_3, color = "Split 3"), show.legend = FALSE) +
  stat_density(aes(x = a_e_3, color = "Split 3"),
               geom = "line", position = "identity") +
  geom_density(aes(x = a_t, color = "True"), show.legend = FALSE) +
  stat_density(aes(x = a_t, color = "True"),
               geom = "line", position = "identity") +
  labs(x = "alpha", y = "Density", title = "alpha Densities", color = "Split") +
  theme_bw()

ggsave(file.path(path, "out", "est_alpha_density.png"), plt)

plt <- ggplot() +
  geom_density(aes(x = b_e_1, color = "Split 1"), show.legend = FALSE) +
  stat_density(aes(x = b_e_1, color = "Split 1"),
               geom = "line", position = "identity") +
  geom_density(aes(x = b_e_2, color = "Split 2"), show.legend = FALSE) +
  stat_density(aes(x = b_e_2, color = "Split 2"),
               geom = "line", position = "identity") +
  geom_density(aes(x = b_e_3, color = "Split 3"), show.legend = FALSE) +
  stat_density(aes(x = b_e_3, color = "Split 3"),
               geom = "line", position = "identity") +
  geom_density(aes(x = b_t, color = "True"), show.legend = FALSE) +
  stat_density(aes(x = b_t, color = "True"),
               geom = "line", position = "identity") +
  labs(x = "beta", y = "Density", title = "beta Densities", color = "Split") +
  theme_bw()

ggsave(file.path(path, "out", "est_beta_density.png"), plt)

#------------------------------------------------------------------------------#
# 5. Implement causal DNN - est_Lambda
#------------------------------------------------------------------------------#
# Project the Hessian onto X to get Lambda
# CM: check the sequence here
cat("Split 2:\n")
est_Lambda2 <- est_Lambda(split = split_2, loss_func = loss_func,
                          y_pred_func = y_pred_func, dnn = dnn_1,
                          arch = c(30, 30, 30, 30, 30, 30), N_epoch = 1000)
cat("Split 1:\n")
est_Lambda1 <- est_Lambda(split = split_1, loss_func = loss_func,
                          y_pred_func = y_pred_func, dnn = dnn_3,
                          arch = c(30, 30, 30, 30, 30, 30), N_epoch = 1000)
cat("Split 3:\n")
est_Lambda3 <- est_Lambda(split = split_3, loss_func = loss_func,
                          y_pred_func = y_pred_func, dnn=dnn_2,
                          arch = c(30, 30, 30, 30, 30, 30), N_epoch = 1000)

#------------------------------------------------------------------------------#
# 6. Implement causal DNN - influence_function
#------------------------------------------------------------------------------#
# 6.1. G(x) = beta(x)
G <- function(split, theta) theta[, 2]

G_if_list_3 <- influence_function(split = split_3, loss_func = loss_func,
                                  y_pred_func = y_pred_func, dnn = dnn_1,
                                  est_Lambda = est_Lambda2, H = G)
G_if_list_2 <- influence_function(split = split_2, loss_func = loss_func,
                                  y_pred_func = y_pred_func, dnn = dnn_3,
                                  est_Lambda = est_Lambda1, H = G)
G_if_list_1 <- influence_function(split = split_1, loss_func = loss_func,
                                  y_pred_func = y_pred_func, dnn = dnn_2,
                                  est_Lambda = est_Lambda3, H = G)

G_if_1 <- G_if_list_1$inf_i # stack
G_if_2 <- G_if_list_2$inf_i
G_if_3 <- G_if_list_3$inf_i

G_hat_est <- mean(c(G_if_1, G_if_2, G_if_3))
G_hat_se <- sqrt((1 / 3) * (var(G_if_1) + var(G_if_2) + var(G_if_3)) / N)

cat(
  round(mean(true_beta %>% as.numeric()), 4),
  round(G_hat_est, 4),
  round(G_hat_se, 4),
  round(G_hat_est - 1.96 * G_hat_se, 4),
  round(G_hat_est + 1.96 * G_hat_se, 4),
  sep = ","
)

# 6.1. H(x) = diff / diff t P {Y == 1}
H <- function(split, theta) {
  z <- split$z
  N <- nrow(z)

  alpha <- theta[,1]$reshape(c(N,1L))
  beta <- theta[,2:(d_out)]$reshape(c(N,d_out-1))
  
  torch_div(
    torch_mul(
      torch_mul(torch_neg(beta), torch_pow(z, torch_neg(torch_add(beta, 1)))),
      torch_exp(torch_neg(alpha))
    ),
    torch_pow(
      torch_add(
        torch_mul(
          torch_pow(z, torch_neg(beta)),
          torch_exp(torch_neg(alpha))
        ),
        1
      ),
      2
    )
  ) %>% return()
}

true_E_H <- torch_div(
  torch_mul(
    torch_mul(
      torch_neg(true_beta), torch_pow(z, torch_neg(torch_add(true_beta, 1)))
    ),
    torch_exp(torch_neg(true_alpha))
  ),
  torch_pow(
    torch_add(
      torch_mul(
        torch_pow(z, torch_neg(true_beta)),
        torch_exp(torch_neg(true_alpha))
      ),
      1
    ),
    2
  )
)

H_if_list_3 <- influence_function(split = split_3, loss_func = loss_func,
                                  y_pred_func = y_pred_func, dnn = dnn_1,
                                  est_Lambda = est_Lambda2, H = H)
H_if_list_2 <- influence_function(split = split_2, loss_func = loss_func,
                                  y_pred_func = y_pred_func, dnn = dnn_3,
                                  est_Lambda = est_Lambda1, H = H)
H_if_list_1 <- influence_function(split = split_1, loss_func = loss_func,
                                  y_pred_func = y_pred_func, dnn = dnn_2,
                                  est_Lambda = est_Lambda3, H = H)

H_if_1 <- H_if_list_1$inf_i # stack
H_if_2 <- H_if_list_2$inf_i
H_if_3 <- H_if_list_3$inf_i

H_hat_est <- mean(c(H_if_1, H_if_2, H_if_3))
H_hat_se <- sqrt((1 / 3) * (var(H_if_1) + var(H_if_2) + var(H_if_3)) / N)

cat(
  round(mean(true_E_H %>% as.numeric()), 4),
  round(H_hat_est, 4),
  round(H_hat_se, 4),
  round(H_hat_est - 1.96 * H_hat_se, 4),
  round(H_hat_est + 1.96 * H_hat_se, 4),
  sep = ","
)
