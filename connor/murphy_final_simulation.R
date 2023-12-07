# murphy_final_simulation.R
# Author: Connacher Murphy
# Simulation for Causal ML Final (Fall 2023)
# Based on methods of Farrell, Liang, and Misra (2021)
# Based on code from Farrell and Misra Causal ML course
# See https://arxiv.org/abs/2010.14694

# CM: add file headings

rm(list = ls()) # clear workspace

library(torch) # load required packages
library(ggplot2)
library(MASS) # for ginv

path <- "~/projects/second-year/causal_ml/final_simulation/" # working directory

set.seed(123) # set seed
torch_manual_seed(123)

#------------------------------------------------------------------------------#
# 1. Generate data
#------------------------------------------------------------------------------#
d_in <- 5 # input dimensionality: dim(x)
dim_z <- 1L # treatment dimensionality: dim(Z)
d_out <- 2 # output dimensionality: dim(theta)

N <- 30000  # number of observations in training set
# CM: increase this after testing

x <- torch_tensor(matrix(rnorm(N * d_in), N, d_in)) # create random data

# Generate alpha(x) = <placeholder>
true_alpha <- x[, 1, NULL] + x[, 2, NULL]

# Generate beta(x) = <placeholder>
true_beta <- x[, 3, NULL] + x[, 4, NULL]

# Assign treatment t(X) = <placeholder>
# t needs to be strictly positive and continuous
# z <- 1 + 0.6 * torch_rand(N, 1L)
# z <- 0.2 + 0.6 * torch_rand(N, 1L)
# z <- torch_pow(torch_randn(N, 1L) + x[, 5, NULL], 2) + 0.5
z <- torch_exp(torch_randn(N, 1L))

# z <- torch_pow(torch_randn(N, 1L), 2) + 1

# z <- x[, 3, NULL] # t in the code is already transformed into log(t)

calc_prob <- function(alpha, beta, z) {
  torch_reciprocal(
    1 + torch_exp(torch_mul(-1, alpha) - torch_mul(beta, torch_log(z)))
  ) %>% return()
}

true_prob <- calc_prob(true_alpha, true_beta, z)

plt <- ggplot() +
  geom_histogram(aes(x = true_prob %>% as.numeric()),
                 color = "deepskyblue3", fill = "deepskyblue") +
  labs(x = "true_prob", y = "Density", title = "Density of true_prob") +
  theme_bw()
ggsave(file.path(path, "out", "density_true_prob.png"), plt)

y <- rbinom(N, 1, prob = as.numeric(true_prob))
y <- torch_tensor( # recast as torch tensor
  as.matrix(y),
  dtype = torch_float()
)

table(as.numeric(y))

calc_loss <- function(y_pred, y, reduction) {
  # nnf_mse_loss(y_pred, y, reduction = reduction) %>% return()
  # y_pred <- torch_where(y_pred == 0, torch_tensor(0.01), y_pred)
  # y_pred <- torch_where(y_pred == 1, torch_tensor(0.99), y_pred)

  loss <- -y * torch_log(y_pred + 0.0001) - (1 - y) *
    torch_log(1 + 0.0001 - y_pred)
  if (reduction == "mean") {
    loss <- loss * (1 / (y %>% length()))
  }
  torch_sum(loss) %>% return()
}

# CM: simple model for testing
# calc_prob <- function(alpha, beta, z) {
#   alpha + torch_mul(beta, z) %>% return()
# }
# true_prob <- calc_prob(true_alpha, true_beta, z) + torch_randn(N, 1)
# true_prob <- calc_prob(true_alpha, true_beta, z)
# y <- torch_tensor( # recast as torch tensor
#   as.matrix(true_prob),
#   dtype = torch_float()
# )

# calc_loss <- function(y_pred, y, reduction) {
#   nnf_mse_loss(y_pred, y, reduction = reduction) %>% return()
# }

#------------------------------------------------------------------------------#
# 2. Create sample splits
#------------------------------------------------------------------------------#
data <- list() # full dataset
data$x <- x
data$y <- y
data$z <- z

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
# 4.1. Implement causal DNN - thetaDNN
#------------------------------------------------------------------------------#
# Crossfits (CM: check crossfit sequences in all sections below):
# 1, 2, 3
# 3, 1, 2
# 2, 3, 1

# CM: switch Adam optimization parameters here
# CM: consider changing architecture
# CM: check loss function
# CM: try some regularization
# Estimate theta with DNN
dnn_1 <- theta_DNN(
  split = split_1, y_pred_func = calc_prob, loss_func = calc_loss,
  N_epoch = 2000, arch = c(20, 20, 20, 20)
)
dnn_2 <- theta_DNN(
  split = split_2, y_pred_func = calc_prob, loss_func = calc_loss,
  N_epoch = 2000, arch = c(20, 20, 20, 20)
)
dnn_3 <- theta_DNN(
  split = split_3, y_pred_func = calc_prob, loss_func = calc_loss,
  N_epoch = 2000, arch = c(20, 20, 20, 20)
)

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
# 4.2. Implement causal DNN - est_Lambda
#------------------------------------------------------------------------------#
# Project the Hessian onto X to get Lambda
# CM: check the sequence here
est_Lambda_1 <- est_Lambda(
  split = split_1, y_pred_func = calc_prob, loss_func = calc_loss, dnn = dnn_3,
  N_epoch = 2000, arch = c(20, 20, 20, 20)
)
est_Lambda_2 <- est_Lambda(
  split = split_2, y_pred_func = calc_prob, loss_func = calc_loss, dnn = dnn_1,
  N_epoch = 2000, arch = c(20, 20, 20, 20)
)
est_Lambda_3 <- est_Lambda(
  split = split_3, y_pred_func = calc_prob, loss_func = calc_loss, dnn = dnn_2,
  N_epoch = 2000, arch = c(20, 20, 20, 20)
)

#------------------------------------------------------------------------------#
# 4.3. Implement causal DNN - influence_function
#------------------------------------------------------------------------------#
# Simple H: H(x) = beta(x)
H <- function(split, theta) theta[, 2]

H_influence_function_1 <- influence_function(
  split = split_1, y_pred_func = calc_prob, loss_func = calc_loss, dnn = dnn_2,
  est_Lambda = est_Lambda_3, H
)
H_influence_function_2 <- influence_function(
  split = split_2, y_pred_func = calc_prob, loss_func = calc_loss, dnn = dnn_3,
  est_Lambda = est_Lambda_1, H
)
H_influence_function_3 <- influence_function(
  split = split_3, y_pred_func = calc_prob, loss_func = calc_loss, dnn = dnn_1,
  est_Lambda = est_Lambda_2, H
)

# Marginal effect
# G <- function(split, theta) {
#   theta[, 2] * torch_pow(split$z, -1) *
#     torch_exp(torch_mul(-1, theta[, 1]) - torch_mul(theta[, 2], torch_log(split$z))) *
#     torch_reciprocal(torch_pow(
#       1 + torch_exp(torch_mul(-1, theta[, 1]) - torch_mul(theta[, 2], torch_log(split$z))),
#       2
#     ))
# }

# CM: linear DNN learning rate adjustment
# CM: check the sequence here
# G_influence_function_1 <- influence_function(
#   split = split_1, y_pred_func = calc_prob, loss_func = calc_loss, dnn = dnn_2,
#   est_Lambda = est_Lambda_3, G
# )
# G_influence_function_2 <- influence_function(
#   split = split_2, y_pred_func = calc_prob, loss_func = calc_loss, dnn = dnn_3,
#   est_Lambda = est_Lambda_1, G
# )
# G_influence_function_3 <- influence_function(
#   split = split_3, y_pred_func = calc_prob, loss_func = calc_loss, dnn = dnn_1,
#   est_Lambda = est_Lambda_2, G
# )

#------------------------------------------------------------------------------#
# 5. Summarize results
#------------------------------------------------------------------------------#
H_ifi_1 <- H_influence_function_1$inf_i
H_ifi_2 <- H_influence_function_2$inf_i
H_ifi_3 <- H_influence_function_3$inf_i

EH_est <- mean(c(H_ifi_1, H_ifi_2, H_ifi_3))
EH_se <- sqrt((1 / 3) * (var(H_ifi_1) + var(H_ifi_2) + var(H_ifi_3)) / N)

EH_est
EH_se

EH_ci_lower <- EH_est - 1.96 * EH_se
EH_ci_upper <- EH_est + 1.96 * EH_se

EH_ci_lower
EH_ci_upper

mean(true_beta) # truth

# G_ifi_1 <- G_influence_function_1$inf_i
# G_ifi_2 <- G_influence_function_2$inf_i
# G_ifi_3 <- G_influence_function_3$inf_i

# EG_est <- mean(c(G_ifi_1, G_ifi_2, G_ifi_3))
# EG_se <- sqrt((1 / 3) * (var(G_ifi_1) + var(G_ifi_2) + var(G_ifi_3)) / N)

# EG_est
# EG_se

# EG_ci_lower <- EG_est - 1.96 * EG_se
# EG_ci_upper <- EG_est + 1.96 * EG_se

# EG_ci_lower
# EG_ci_upper
