linear_DNN <- function(x, y, arch = c(20, 20), N_epoch = 1000, .seed = 1233) {
  d_in <- ncol(x) # dim(x)
  d_out <- ncol(y) # dim(y)
  N <- nrow(x) # number of observations in split
  
  set.seed(.seed) # set seed
  torch_manual_seed(.seed)
  
  # Construct NN
  model <- nn_sequential(nn_linear(d_in, arch[1]), nn_relu())
  j <- 1 # for a single layer
  if (length(arch) > 1) { # loop through layers in arch
    for (j in 2:length(arch)) {
      model$add_module( # linear combination
        name = paste0("layer_", j - 1), # CM: why name with j - 1 instead of j?
        module = nn_linear(arch[j - 1], arch[j])
      )
      model$add_module( # ReLU activation
        name = paste0("relu_", j - 1), # CM: why name with j - 1 instead of j?
        nn_relu() # nn_sigmoid()
      )
      # model$add_module( # testing dropout
      #   name = paste0("dropout_", j - 1), # CM: why name with j - 1 instead of j?
      #   nn_dropout(0.2)
      # )
    }
  }
  model$add_module("xb", nn_linear(arch[j], d_out)) # output layer
  
  # learning_rate <- 0.01
  # optimizer <- optim_adam(model$parameters, lr = learning_rate)
  learning_rate <- 1e-3 # optimization details (CM: switch to argument)
  optimizer <- optim_adam(model$parameters, lr = learning_rate, weight_decay = 1e-5)
  
  # Initialize weights (with implicit regularization)
  for (i in seq_along(model$parameters)) {
    nn_init_normal_(model$parameters[[i]], 0, .1)
  }
  
  # Prepare for training loop
  interval <- N_epoch / 100 # for progress bar
  cat("\nTraining DNN...\n")
  prog_bar <- txtProgressBar(min = 1, max = 100, style = 3)
  pct_prog <- 0
  # loss.store <- NULL

  for (e in 1:N_epoch) { # training loop
    y_pred <- model(x) # forward pass

    loss <- nnf_mse_loss(y_pred, y, reduction = "mean") # MSE loss
    # loss <- 0.5 * nnf_mse_loss(y_pred, y, reduction = "mean") # MSE loss

    if (is.nan(loss %>% as.numeric())) {
      stop("NaN loss detected :(")
    }

    optimizer$zero_grad() # backpropagation
    loss$backward()

    optimizer$step() # use the optimizer to update model parameters

    if(e %% interval == 0) { # progress
      pct_prog <- pct_prog + 1
      setTxtProgressBar(prog_bar, pct_prog)
    }
    # loss.store <- c(loss.store, as.numeric(loss))
  }
  cat("\n")
  list(model = model) # just need the trained DNN
}
