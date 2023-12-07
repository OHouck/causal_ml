# Automatic Inference on Causal Deepnets
# CML 2023
# Instructors: Max Farrell & Sanjog Misra
# Code to implement second stage of CausalDNN 
# https://arxiv.org/abs/2010.14694
# Projects Hessian of loss function on to X
# Returns projection functions projH 
# dat must be alist with three elements
# x: Covariates
# y: Outcomes
# z: treatments
# Each must be a torch tensor (with col dimension defined) 
# Code is provided for teaching and illustration purposes. Use at your own risk.
# The code has not been designed or optimized for research purposes.
# Last Edited: Nov 13, 2023 by Sanjog Misra

makeLam = function(dat,dnn,arch=c(5,5),NEPOCH=200){
  
  mod = dnn$model # Get ab(x)
  
  # Data elements
  x = dat$x
  z = dat$z
  y = dat$y
  n = nrow(dat$x)
  # Constants
  d_in <- ncol(x)
  d_z <- ncol(z)
  d_out <- d_z + 1  # dim(b)+dim(a)

  # Extract Parameters   
  ab = mod(x)
  alpha=ab[,1]$reshape(c(n,1L))
  beta=ab[,2:d_out]$reshape(c(n,d_out-1))
  
  # Retain grads
  alpha$retain_grad()
  beta$retain_grad()
  ab$retain_grad()
  
  # Prediction and Loss
  y_pred <- alpha+torch_sum(beta*z,dim = 2,keepdim = TRUE)
  loss <- .5*nnf_mse_loss(y_pred, y, reduction = "sum") # again note sum
  
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
  
  # Let's Project Hessian columns on to X
  # And store models
  
  cat("\n working on projections...\n")
  projH = list()
  
  # For example:
  # Y: h1,h2,h3,h4 
  # X: X
  # Y = f_dnn(x) + e 
  # h1 = f_dnn(X)
  # h2 = f_dnn(X) ...
  # E(L_theta_theta | X) is same as E(Y|X)
  
  for(j in 1:ncol(hess.vec)){
    # Linear Regression option
    # projH[[j]] = lm(as.matrix(hess.vec[,j])~as.matrix(x))
    Hy = torch_tensor(as.matrix(hess.vec[,j]))
    projH[[j]]=linDNN(x=x,y=Hy,arch=arch,NEPOCH=200)
  }
  cat("\n done.\n")
  
  
  return(projH)
}
