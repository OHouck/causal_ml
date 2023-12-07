est_Lambda <- function(
  split, y_pred_func, loss_func, dnn, arch = c(5, 5), N_epoch = 200
) {
  model <- dnn$model

  x <- split$x # extract tensors from split
  y <- split$y
  z <- split$z

  d_in <- ncol(x) # dim(x)
  d_z <- ncol(z) # dim(z)
  d_out <- 2  # dim(theta)

  N <- nrow(x) # number of observations in split

  theta <- model(x) # extract parameters
  alpha <- theta[, 1]$reshape(c(N, 1L))
  beta <- theta[, 2:(d_out)]$reshape(c(N, d_out - 1))

  theta$retain_grad() # retain gradients
  alpha$retain_grad()
  beta$retain_grad()

  y_pred <- y_pred_func(alpha, beta, z) # Prediction and loss
  # Prediction and Loss
  # y_pred <- alpha + torch_sum(beta * z, dim = 2, keepdim = TRUE)
  # loss <- 0.5 * nnf_mse_loss(y_pred, y, reduction = "sum")
  # loss <- 0.5 * loss_func(y_pred, y, reduction = "sum")
  loss <- loss_func(y_pred, y, reduction = "sum")
  
  if (is.nan(loss %>% as.numeric())) {
    stop("NaN loss detected :(")
  }

  theta_grad <- autograd_grad( # compute gradient
    loss, theta, create_graph = TRUE, retain_graph = TRUE
  )

  K <- dim(theta_grad[[1]])[2]
  theta_grad_sum <- theta_grad[[1]]$sum(1)

  h_tmp <- list() # save the Hessian as a flat vector
  for (j in 1:K) {
    h_tmp[[j]] <- autograd_grad(
      theta_grad_sum[j], theta, retain_graph = TRUE
    )[[1]]
  }
  hess_vec = torch_cat(h_tmp, 2)

  cat("\nProjecting Hessian onto X\n")

  projections <- list()

  for (j in 1:ncol(hess_vec)) {
    # Linear regression:
    # projections[[j]] <- lm(as.matrix(hess_vec[, j]) ~ as.matrix(x))

    # DNN:
    y_H <- torch_tensor(as.matrix(hess_vec[, j]))
    projections[[j]] <- linear_DNN(
      x = x, y = y_H, arch = arch, N_epoch = N_epoch
    )
  }
  cat("\n")

  return(projections)
}
