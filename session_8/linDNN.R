# Helper function
#Runs Linear regression y=f(x)+e
# f(x) is approximated by a DNN
# Sanjog Misra

linDNN=function(x=x,y=y,arch=c(20,20),NEPOCH=1000,.seed=1234){
  # input dimensionality (number of input features)
  d_in <- ncol(x)
  # Number of outcomes
  d_out <- ncol(y) 
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
  model$add_module("xb",nn_linear(arch[j],d_out))  
  
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
    y_pred <- model(x)
    loss <- nnf_mse_loss(y_pred, y, reduction = "mean")
    
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
  
  list(model=model,yhat=y_pred,loss.stor=loss.stor)
  
}





