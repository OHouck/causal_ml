# Automatic Inference on Causal Deepnets
# CML 2023
# Instructors: Max Farrell & Sanjog Misra
# Code to prep for third stage of CausalDNN
# Returns Objects needed for IF computation
# dat must be alist with three elements
# Each must be a torch tensor (with col dimension defined) 
# x: Covariates
# y: Outcomes
# z: treatments
# H: function of interest. Must take in ab and return a torch tensor (without changing graph) 
# Code is provided for teaching and illustration purposes. Use at your own risk.
# The code has not been designed or optimized for research purposes.
# Last Edited: Nov 13, 2023 by Sanjog Misra
# and provides all components needed for inference

proc_res = function(dat,dnn,lproj,H){
  
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
  
  # Now let's use projections  
  projH = lproj
  hess.vec = NULL
  for(k in 1:length(projH)){
    hess.vec = cbind(hess.vec,as.matrix(projH[[k]]$model(x)))
  }
  
  # Automatically differentiate H construct
  # Function of interest (ATE here)
  Hi = H(ab,dat) # What is this?
  
  H = Hi$sum() # Sum trick again
  Hab = autograd_grad(H,ab) # H_theta
  Hab2 = (as.array(Hab[[1]]))
  n=nrow(ab)
  
  # Extract hess.vec
  h1 = hess.vec
  gg = as.matrix(abg[[1]])
  
  # Convert in Matrix form
  V = t(apply(h1,1,function(li) c(ginv(matrix(li,2,2)))))
  # ginv: PseudoInverse
  # (H+eta*I)^(-1) neq H^-1
  
  # Invert for each row
  lami=list()
  for(j in 1:n) lami[[j]] = matrix(V[j,],2,2) 

    # Compute IF
  auto.if = as.numeric(Hi)+sapply(1:n,function(j) Hab2[j,]%*%lami[[j]]%*%t(gg)[,j])
  
  
  return(list(auto.if=auto.if,plugin=as.numeric(Hi)))

  }
