# Automatic Inference on Causal Deepnets
# CML 2023
# Instructors: Max Farrell & Sanjog Misra
# Code to estimate first stage of CausalDNN
# https://arxiv.org/abs/2010.14694

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

cdnn<-function(dat1,arch=c(20,20),NEPOCH=1000,.seed = 1234)
{
  x = dat1$x
  y = dat1$y
  z = dat1$z
  
  # input dimensionality (number of input features)
  d_in <- ncol(x)
  
  # Treatments
  d_z <- ncol(z)
  
  # Number of parameters
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
  model$add_module("ab",nn_linear(arch[j],d_out))  
  
  # Learning framework
  # for ADAM, need to choose a reasonable learning rate
  learning_rate <- 0.01
  optimizer <- optim_adam(model$parameters, lr = learning_rate)
  
  # Initialize
  for(i in seq_along(model$parameters))
  {
    nn_init_normal_(model$parameters[[i]],0,.1)
  }
  
  # Training Loop
  intv = NEPOCH/100
  cat("\n Begining training...\n")
  pb <- txtProgressBar(min=1,max=100,style=3)
  pct=0
  loss.stor = NULL
  st = proc.time()
  
  # Training loop
  for (t in 1:NEPOCH) {
    ### -------- Forward pass -------- 
    ### Causal Model 
    ab <- model(x)
    alpha=ab[,1]$reshape(c(n,1L))
    beta=ab[,2:(d_out)]$reshape(c(n,d_out-1))
    # Notice anything different?
    y_pred <- alpha+torch_sum(beta*z,dim = 2,keepdim = TRUE)
    loss <- nnf_mse_loss(y_pred, y, reduction = "mean")
    
    # Change loss as we discussed in class
    # e.g. binary_nll2 
    # Will have to change the loss in all places/steps. 
    # Code can be adapted to allow for a different loss.
    
    # -------- Backpropagation -------- 
    # Need to zero out the gradients before the backward pass
    optimizer$zero_grad()
    
    # gradients are computed on the loss tensor
    loss$backward()
    
    # -------- Update weights -------- 
    # use the optimizer to update model parameters
    optimizer$step()
    
    # progress
    if(t%%intv==0) {pct=pct+1; setTxtProgressBar(pb, pct)}
    
    loss.stor = c(loss.stor,as.numeric(loss)) 
  }
  
  cat("/n")
  
  # We will need some more "stuff"
  # Prediction and Loss
  y_pred <- alpha+torch_sum(beta*z,dim = 2,keepdim = TRUE)
  loss <- .5*nnf_mse_loss(y_pred, y, reduction = "sum") # note .5 and sum (why?)
  
  
  ## Compute gradient (use sum trick)
  abg = autograd_grad(loss,ab,create_graph = TRUE,retain_graph = TRUE)
  
  # Sum Trick
  K=dim(abg[[1]])[2]
  abg_sum = abg[[1]]$sum(1) # Why do this?
  
  # We will save the Hessian as a flat vector
  h.tmp = list()
  for(j in 1:K){
    h.tmp[[j]]=autograd_grad(abg_sum[j],ab,retain_graph = TRUE)[[1]]
  }
  
  # Concat the results 
  hess.vec = torch_cat(h.tmp,2)
  # Return list
  list(model=model,ab = ab,loss.stor=loss.stor,hess.vec=hess.vec)
}
