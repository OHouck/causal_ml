influence_function <- function(
  split, y_pred_func, loss_func, dnn, est_Lambda, H
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

  # Prediction and Loss
  # y_pred <- alpha + torch_sum(beta * z, dim = 2, keepdim = TRUE)
  y_pred <- y_pred_func(alpha, beta, z) # Prediction and loss
  # loss <- 0.5 * loss_func(y_pred, y, reduction = "sum")
  loss <- loss_func(y_pred, y, reduction = "sum")
  # loss <- 0.5 * nnf_mse_loss(y_pred, y, reduction = "sum")

  theta_grad <- autograd_grad( # compute gradient
    loss, theta, create_graph = TRUE, retain_graph = TRUE
  )
  theta_grad2 <- as.matrix(theta_grad[[1]]) # useful to store as a matrix

  hess_vec <- NULL # store the Hessian as a flat vector
  for(k in 1:length(est_Lambda)){
    hess_vec <- cbind(hess_vec, as.matrix(est_Lambda[[k]]$model(x)))
  }

  H_i <- H(split, theta) # Differentiate H with respect to theta
  H <- H_i$sum()
  H_theta <- autograd_grad(H, theta)
  H_theta2 <- as.array(H_theta[[1]]) # useful to store as an array
  m <- nrow(theta)

  # (Pseudo)invert the Hessian
  L <- t(apply(hess_vec, 1, function(l) c(ginv(matrix(l, 2, 2)))))
  Lambda_inv <- list()
  for (j in 1:m) Lambda_inv[[j]] <- matrix(L[j, ], 2, 2)

  # Compute influence function
  inf_i <- as.numeric(H_i) + # plugin
    sapply( # correction
      1:m, function(j) H_theta2[j, ] %*% Lambda_inv[[j]] %*% t(theta_grad2)[, j]
    )
  # return(list(auto.if=auto.if,plugin=as.numeric(Hi)))
  list(inf_i = inf_i)
}
