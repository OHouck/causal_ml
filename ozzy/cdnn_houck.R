
# Automatic Inference on Causal Deepnets
# CML 2023
# Authors: Ozzy modified code by Max Farrell & Sanjog Misra
# Code to estimate first stage of CausalDNN
# https://arxiv.org/abs/2010.14694

# Ozzy's changes, makes learning rate, y_pred_func, and loss functions parameters

# Returns parameter functions (model(x)=theta(x)) in "model" object
# The architecture (all layers are relu)
# dat1 must be alist with three elements
# x: Covariates
# y: Outcomes
# z: treatments
# Each must be a torch tensor (with col dimension defined) 
# arch: defines the architecture of the DNN, number of nodes in each layer
# NEPOCH: Number of EPOCHS
# This code does not do minibatching. Add if needed.
# Code is provided for teaching and illustration purposes. Use at your own risk.
# The code has not been designed or optimized for research purposes.
# Last Edited: Nov 13, 2023 by Sanjog Misra

cdnn<-function(split, y_pred_func, loss_func, learn_rate, weight_decay,
     arch=c(20,20),NEPOCH=1000,.seed = 123)
{
  x = split$x
  y = split$y
  z = split$z
  
  # input dimensionality (number of input features)
  d_in <- ncol(x)
  
  # Treatments
  d_z <- ncol(z)
  
  # Number of parameters (should be 2 for alpha and beta in this context)
  d_out <- d_z + 1  # dim(b)+dim(a)
  
  # number of observations in full data set
  n <- nrow(x)
  
  # Set a seed (mostly for replication)
  set.seed(.seed)
  torch_manual_seed(.seed)
  
  # Model constructor
  model <- nn_sequential(nn_linear(d_in, arch[1]),nn_relu())
  j=1 # in case single layer
  # Loop through architecture 
  if(length(arch)>1){
    for(j in 2:length(arch)){
      model$add_module(name = paste("layer_",j-1),module=nn_linear(arch[j-1],arch[j]))  
      model$add_module(name = paste0("relu_",j-1),nn_relu())  
    }
  }
  
  # (Parameter) Output layer
  model$add_module("theta", nn_linear(arch[j],d_out))  
  
  # Learning framework
  # for ADAM, need to choose a reasonable learning rate
  # Conor adds a weight decay parameter XX
  optimizer <- optim_adam(model$parameters, lr = learn_rate, weight_decay = weight_decay)
  
  # Initialize
  for(i in seq_along(model$parameters))
  {
    nn_init_normal_(model$parameters[[i]],0,.1)
  }
  
  # Training Loop
  intv  <- NEPOCH/100
  cat("\n Begining training...\n")
  progress_bar <- txtProgressBar(min=1,max=100,style=3)
  progress_pct  <- 0
  loss.stor  <-  NULL
  st = proc.time()
  
  # Training loop
  for (t in 1:NEPOCH) {
    ### -------- Forward pass -------- 
    ### Causal Model 
    theta <- model(x)
    alpha=theta[,1]$reshape(c(n,1L))
    beta=theta[,2:(d_out)]$reshape(c(n,d_out-1))

    # alpha and beta are tensor objects 
    y_pred <- y_pred_func(alpha, beta, z)

    # loss <- nnf_mse_loss(y_pred, y, reduction = "mean")
    loss <- loss_func(y_pred, y)

    # check for NAs
    if (is.nan(loss %>% as.array())) {
        print("Loss is NaN")
        break
    }
    
    # -------- Backpropagation -------- 
    # Need to zero out the gradients before the backward pass
    optimizer$zero_grad()
    
    # gradients are computed on the loss tensor
    loss$backward()
    
    # -------- Update weights -------- 
    # use the optimizer to update model parameters
    optimizer$step()
    
    # progress
    if(t%%intv==0) {
        progress_pct = progress_pct + 1
        setTxtProgressBar(progress_bar, progress_pct)
    }
    
    loss.stor = c(loss.stor,as.numeric(loss)) 
  }
  
  cat("/n")

  # just need to return the model
  list(model = model) 


 # Extra stuff Ozzy doesn't want now 
  # We will need some more "stuff"
  # Prediction and Loss
#   y_pred <- alpha+torch_sum(beta*z,dim = 2,keepdim = TRUE)
#   loss <- .5*nnf_mse_loss(y_pred, y, reduction = "sum") # note .5 and sum (why?)
  
  
#   ## Compute gradient (use sum trick)
#   abg = autograd_grad(loss,ab,create_graph = TRUE,retain_graph = TRUE)
  
#   # Sum Trick
#   K=dim(abg[[1]])[2]
#   abg_sum = abg[[1]]$sum(1) # Why do this?
  
#   # We will save the Hessian as a flat vector
#   h.tmp = list()
#   for(j in 1:K){
#     h.tmp[[j]]=autograd_grad(abg_sum[j],ab,retain_graph = TRUE)[[1]]
#   }
  
#   # Concat the results 
#   hess.vec = torch_cat(h.tmp,2)
#   # Return list
#   list(model=model,ab = ab,loss.stor=loss.stor,hess.vec=hess.vec)
}
