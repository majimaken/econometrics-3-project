

data_function<-function(x,lag_vec,in_out_sample_separator)
{
  # 3.b
  data_mat<-x
  for (j in 1:length(lag_vec))
    data_mat<-cbind(data_mat,lag(x,k=lag_vec[j]))
  # Check length of time series before na.exclude
  dim(data_mat)
  data_mat<-na.exclude(data_mat)
  # Check length of time series after removal of NAs
  dim(data_mat)
  head(data_mat)
  tail(data_mat)
  
  #--------------------------------------------------------------------
  # 3.c&d Specify in- and out-of-sample episodes
  
  target_in<-data_mat[paste("/",in_out_sample_separator,sep=""),1]
  tail(target_in)
  explanatory_in<-data_mat[paste("/",in_out_sample_separator,sep=""),2:ncol(data_mat)]
  tail(explanatory_in)
  
  target_out<-data_mat[paste(in_out_sample_separator,"/",sep=""),1]
  head(target_out)
  tail(target_out)
  explanatory_out<-data_mat[paste(in_out_sample_separator,"/",sep=""),2:ncol(data_mat)]
  head(target_out)
  tail(explanatory_out)
  
  train<-cbind(target_in,explanatory_in)
  test<-cbind(target_out,explanatory_out)
  head(test)
  tail(test)
  nrow(test)
  
  # Scaling data for the NN
  maxs <- apply(data_mat, 2, max) 
  mins <- apply(data_mat, 2, min)
  # Transform data into [0,1]  
  scaled <- scale(data_mat, center = mins, scale = maxs - mins)
  
  apply(scaled,2,min)
  apply(scaled,2,max)
  #-----------------
  # 4.b
  # Train-test split
  train_set <- scaled[paste("/",in_out_sample_separator,sep=""),]
  test_set <- scaled[paste(in_out_sample_separator,"/",sep=""),]
  
  train_set<-as.matrix(train_set)
  test_set<-as.matrix(test_set)
  
  
  return(list(data_mat=data_mat,target_in=target_in,target_out=target_out,explanatory_in=explanatory_in,explanatory_out=explanatory_out,train_set=train_set,test_set=test_set))
}




estimate_nn<-function(train_set,number_neurons,data_mat,test_set,f)
{
  nn <- neuralnet(f,data=train_set,hidden=number_neurons,linear.output=T)
  
  
  # In sample performance
  predicted_scaled_in_sample<-nn$net.result[[1]]
  # Scale back from interval [0,1] to original log-returns
  predicted_nn_in_sample<-predicted_scaled_in_sample*(max(data_mat[,1])-min(data_mat[,1]))+min(data_mat[,1])
  # In-sample MSE
  MSE.in.nn<-mean(((train_set[,1]-predicted_scaled_in_sample)*(max(data_mat[,1])-min(data_mat[,1])))^2)
  
  # Out-of-sample performance
  # Compute out-of-sample forecasts
  pr.nn <- compute(nn,test_set[,2:ncol(test_set)])
  predicted_scaled<-pr.nn$net.result
  # Results from NN are normalized (scaled)
  # Descaling for comparison
  predicted_nn <- predicted_scaled*(max(data_mat[,1])-min(data_mat[,1]))+min(data_mat[,1])
  test.r <- test_set[,1]*(max(data_mat[,1])-min(data_mat[,1]))+min(data_mat[,1])
  # Calculating MSE
  MSE.out.nn <- mean((test.r - predicted_nn)^2)
  
  # Compare in-sample and out-of-sample
  MSE_nn<-c(MSE.in.nn,MSE.out.nn)
  return(list(MSE_nn=MSE_nn,predicted_nn=predicted_nn,predicted_nn_in_sample=predicted_nn_in_sample))
  
}







neural_net_predict<-function(data_obj,number_neurons,period_sharpe)
{
  train_set<-data_obj$train_set
  test_set<-data_obj$test_set
  target_out<-data_obj$target_out
  target_in<-data_obj$target_in
  data_mat<-data_obj$data_mat
  
  n <- colnames(train_set)
  # Model: target is current bitcoin, all other variables are explanatory  
  f <- as.formula(paste(colnames(train_set)[1]," ~", paste(n[!n %in% colnames(train_set)[1]], collapse = " + ")))
  
  corr_vec<-sharpe_nn<-sharpe_nn_in<-1:anz_real
  MSE_mat<-matrix(ncol=2,nrow=anz_real)
  colnames(MSE_mat)<-c("In sample MSE","Out sample MSE")
  #-----------------------
  # 7.b
  
  pb <- txtProgressBar(min = 1, max = anz_real, style = 3)
  
  # One could try alternative set.seeds and/or larger anz_real
  set.seed(0)
  for (i in 1:anz_real)#i<-1
  {
    
    nn.obj<-estimate_nn(train_set,number_neurons,data_mat,test_set,f)
    
    predicted_nn<-nn.obj$predicted_nn
    if (i==1)
    {
      predicted_mat<-predicted_nn
    } else
    {
      predicted_mat<-cbind(predicted_mat,predicted_nn)
    }
    predicted_nn_in_sample<-nn.obj$predicted_nn_in_sample
    MSE_mat[i,]<-nn.obj$MSE_nn
    # Go long or short depending on sign of forecast
    #   We do not need to lag the signal here since the forecast is based on (already) lagged data 
    perf_nn<-(sign(predicted_nn))*target_out
    perf_nn_in<-(sign(predicted_nn_in_sample))*target_in
    if (i==1)
    {  
      perf_nn_mat<-perf_nn
      perf_nn_mat_in<-perf_nn_in
    } else
    {
      perf_nn_mat<-cbind(perf_nn_mat,perf_nn)
      perf_nn_mat_in<-cbind(perf_nn_mat_in,perf_nn_in)
    }
    
    sharpe_nn[i]<-sqrt(period_sharpe)*mean(perf_nn,na.rm=T)/sqrt(var(perf_nn,na.rm=T))
    sharpe_nn_in[i]<-sqrt(period_sharpe)*mean(perf_nn_in,na.rm=T)/sqrt(var(perf_nn_in,na.rm=T))
    
    setTxtProgressBar(pb, i)
  }
  close(pb)
  return(list(MSE_mat=MSE_mat,perf_nn_mat=perf_nn_mat,sharpe_nn=sharpe_nn,predicted_mat=predicted_mat))
}  











# This function fits a FF-net with two hidden layers and computes unadjusted out-of-sample MSE
estimate_keras_ff_regression<-function(explanatory_train,target_train,explanatory_test,target_test,number_neurons,data_mat,test_set,batch_size,epochs)
{
  
  
  model_reg <- keras_model_sequential() 
  model_reg %>% 
    layer_dense(units = number_neurons[1], activation = "relu", input_shape = c(ncol(explanatory_train))) %>% 
    #    layer_dropout(rate = 0.4) %>% 
    layer_dense(units = number_neurons[2], activation = "relu") %>%
    #    layer_dropout(rate = 0.3) %>%
    layer_dense(units = 1, activation = "sigmoid")
  # Regression: MSE measure  
  model_reg %>% compile(
    loss = "mse",
    optimizer = "adam",
    metrics = c("mse"))#select c("accuracy"): for classification problems
  
  summary(model_reg)
  
  model_reg %>% fit(
    explanatory_train, target_train,
    batch_size = batch_size,
    epochs = epochs,
    validation_data = list(explanatory_test, target_test))
  
  scores_reg <- model_reg %>% evaluate(
    explanatory_test, target_test,
    batch_size = batch_size
  )
  perf_mat<-scores_reg$mean_squared_error
  
  predicted_keras_adjusted<-predict(model_reg,explanatory_test)
  
  predicted_keras<-predicted_keras_adjusted*(max(data_mat[,1])-min(data_mat[,1]))+min(data_mat[,1])
  
  test.r <- target_test*(max(data_mat[,1])-min(data_mat[,1]))+min(data_mat[,1])
  # Calculating MSE
  MSE.keras <- mean((test.r -predicted_keras )^2)
  
  
  return(list(MSE.keras=MSE.keras,predicted_keras=predicted_keras))
  
}









keras_feedforward_predict<-function(data_obj,number_neurons,period_sharpe,epochs)
{
  train_set<-data_obj$train_set
  test_set<-data_obj$test_set
  target_out<-data_obj$target_out
  target_in<-data_obj$target_in
  data_mat<-data_obj$data_mat
  
  target_train<-train_set[,1]
  explanatory_train<-as.matrix(train_set[,2:ncol(train_set)])
  target_test<-test_set[,1]
  explanatory_test<-as.matrix(test_set[,2:ncol(train_set)])
  
  
  MSE_mat<-matrix(ncol=1,nrow=anz_real)
  colnames(MSE_mat)<-"Out sample MSE"
  
  pb <- txtProgressBar(min = 1, max = anz_real, style = 3)
  batch_size <- nrow(explanatory_train)
  sharpe_keras<-1:anz_real
  # One could try alternative set.seeds and/or larger anz_real
  #  use_session_with_seed(1, disable_gpu = FALSE, disable_parallel_cpu = FALSE)
  for (i in 1:anz_real)#i<-1
  {
    
    
    # Note that loss and mean-squared error are identical in our case (since we selected MSE as performance measure)
    #   In-sample (training) and out-of-sample (validation) MSEs
    #   Slight overfitting visible for epochs>100
    fit_keras_net<-estimate_keras_ff_regression(explanatory_train,target_train,explanatory_test,
                                                target_test,number_neurons,data_mat,test_set,batch_size,epochs)
    
    predicted_keras<-fit_keras_net$predicted_keras
    if (i==1)
    {
      predicted_mat<-predicted_keras
    } else
    {
      predicted_mat<-cbind(predicted_mat,predicted_keras)
    }
    MSE_mat[i,]<-fit_keras_net$MSE.keras
    # Go long or short depending on sign of forecast
    #   We do not need to lag the signal here since the forecast is based on (already) lagged data 
    
    setTxtProgressBar(pb, i)
  }
  close(pb)
  return(list(MSE_mat=MSE_mat,predicted_mat=predicted_mat))
}  





# This function fits a FF-net with two hidden layers and computes unadjusted out-of-sample MSE
estimate_mxnet_feedforward<-function(explanatory_train,target_train,explanatory_test,target_test,number_neurons,data_mat,test_set,batch_size,epochs)
{
  
  data <- mx.symbol.Variable("data")
  
  # Optimization is not useful: optimal mean (explanatory data is ignored)
  # First hidden layer  
  fc1 <- mx.symbol.FullyConnected(data, num_hidden=number_neurons[1]) 
  #act1 <- mx.symbol.Activation(fc1, name="relu1", act_type="relu") 
  act1 <- mx.symbol.Activation(fc1, name="sigmoid", act_type="sigmoid") 
  # Second hidden layer
  fc2 <- mx.symbol.FullyConnected(act1, name="fc2", num_hidden=number_neurons[2]) 
  #act2 <- mx.symbol.Activation(fc2, name="relu2", act_type="relu")
  act2 <- mx.symbol.Activation(fc2, name="sigmoid", act_type="sigmoid")
  # Output layer: regression i.e. one single output neuron
  fc3 <- mx.symbol.FullyConnected(act2, name="fc3", num_hidden=1) 
  lro <- mx.symbol.LinearRegressionOutput(fc3)
  # Output layer: binary classification therefore num_hidden=2 
  #  fc3 <- mx.symbol.FullyConnected(act2, name="fc3", num_hidden=2) 
  #  lro <- mx.symbol.SoftmaxOutput(fc3, name="sm") 
  
  
  #What matters for a regression task is mainly the last function. It enables the new network 
  # to optimize for squared loss. 
  # See below for alternative error measures
  mx_mse <- mx.model.FeedForward.create(lro,  X=explanatory_train, y=target_train,
                                        ctx=mx.cpu(), num.round=epochs, array.batch.size=batch_size,
                                        learning.rate=1e-02, momentum=0.5, 
                                        eval.metric=mx.metric.rmse)
  
  # Hier ein kurzer Einschub zum Fehlermass  
  if (F)
  {
    # Alternative Fehlermasse
    #   1. Klassifikation    
    eval.metric=mx.metric.accuracy
    #   2. Custom: eigenes Fehlermass verwenden     
    demo.metric.mae <- mx.metric.custom("mae", function(label, pred) {
      res <- mean(abs(label-pred))
      return(res)
    }) 
    eval.metric=demo.metric.mae 
  }
  #
  
  preds_scaled = predict(mx_mse, explanatory_test)
  
  predicted_mxnet<-preds_scaled[1,]*(max(data_mat[,1])-min(data_mat[,1]))+min(data_mat[,1])
  
  
  test.r <- target_test*(max(data_mat[,1])-min(data_mat[,1]))+min(data_mat[,1])
  # Calculating MSE
  MSE.mxnet <- mean((test.r - predicted_mxnet)^2)
  
  return(list(MSE.mxnet=MSE.mxnet,predicted_mxnet=predicted_mxnet))
  
}









mxnet_feedforward_predict<-function(data_obj,number_neurons,period_sharpe,epochs)
{
  train_set<-data_obj$train_set
  test_set<-data_obj$test_set
  target_out<-data_obj$target_out
  target_in<-data_obj$target_in
  data_mat<-data_obj$data_mat
  
  target_train<-train_set[,1]
  explanatory_train<-as.matrix(train_set[,2:ncol(train_set)])
  target_test<-test_set[,1]
  explanatory_test<-as.matrix(test_set[,2:ncol(train_set)])
  
  
  MSE_mat<-matrix(ncol=1,nrow=anz_real)
  colnames(MSE_mat)<-"Out sample MSE"
  
  pb <- txtProgressBar(min = 1, max = anz_real, style = 3)
  batch_size <- nrow(explanatory_train)
  sharpe_mxnet<-1:anz_real
  # One could try alternative set.seeds and/or larger anz_real
  mx.set.seed(0)
  for (i in 1:anz_real)#i<-1
  {
    
    
    # Note that loss and mean-squared error are identical in our case (since we selected MSE as performance measure)
    #   In-sample (training) and out-of-sample (validation) MSEs
    #   Slight overfitting visible for epochs>100
    fit_mxnet<-estimate_mxnet_feedforward(explanatory_train,target_train,explanatory_test,target_test,number_neurons,data_mat,test_set,batch_size,epochs)
    
    
    predicted_mxnet<-fit_mxnet$predicted_mxnet
    if (i==1)
    {
      predicted_mat<-predicted_mxnet
    } else
    {
      predicted_mat<-cbind(predicted_mat,predicted_mxnet)
    }
    MSE_mat[i,]<-fit_mxnet$MSE.mxnet
    # Go long or short depending on sign of forecast
    #   We do not need to lag the signal here since the forecast is based on (already) lagged data 
    
    setTxtProgressBar(pb, i)
  }
  close(pb)
  return(list(MSE_mat=MSE_mat,predicted_mat=predicted_mat))
}  





trading_func<-function(predict.obj,center_forecast,data_obj,period_sharpe)
{
  
  predicted_mat<-predict.obj$predicted_mat
  target_out<-data_obj$target_out
  sharpe_vec<-rep(NA,anz_real)
  
  for (i in 1:anz_real)#i<-1
  {
    
    
    # Note that loss and mean-squared error are identical in our case (since we selected MSE as performance measure)
    #   In-sample (training) and out-of-sample (validation) MSEs
    #   Slight overfitting visible for epochs>100
    # Go long or short depending on sign of forecast
    #   We do not need to lag the signal here since the forecast is based on (already) lagged data 
    if (center_forecast)
    {
      perf<-(sign(predicted_mat[,i]-mean(predicted_mat[,i])))*target_out
    } else
    {
      perf<-(sign(predicted_mat[,i]))*target_out
    }
    if (i==1)
    {  
      perf_mat<-perf
    } else
    {
      perf_mat<-cbind(perf_mat,perf)
    }
    
    sharpe_vec[i]<-sqrt(period_sharpe)*mean(perf,na.rm=T)/sqrt(var(perf,na.rm=T))
    
  }
  return(list(perf_mat=perf_mat,sharpe_vec=sharpe_vec))
} 






# Keras lstm fit
#   The network has three hidden layers

estimate_keras_lstm_func<-function(x_train,y_train,x_test,y_test,number_neurons,data_mat,batch_size,epochs)
{
  
  batch_len<-dim(y_train)[1]
  
  # See e.g. https://keras.rstudio.com/reference/layer_lstm.html  
  model <- keras_model_sequential() 
  model %>% 
    layer_lstm(units = number_neurons[1], return_sequences = TRUE, input_shape = c(dim(x_train)[2], 1)) %>% 
    layer_lstm(units = number_neurons[2], return_sequences = TRUE) %>% 
    layer_lstm(units = number_neurons[3]) %>% # return a single vector dimension 32
    layer_dense(units = 1, activation = "sigmoid") %>% # Alternatives: "relu" or "softmax"
    compile(
      loss = "mse",  
      optimizer = "adam", 
      metrics = c("mse") 
    )
  
  
  # use_session_with_seed(42)
  model %>% fit( 
    x_train, y_train, batch_size = batch_len, epochs = epochs, validation_data = list(x_test, y_test)
  )
  
  scores <- model %>% evaluate(
    x_test, y_test)#,
  #    batch_size = dim(x_train)[2]
  #  )
  
  cat("Test loss:", scores[[1]])
  cat("Test mse", scores[[2]])
  
  
  
  # These two are identical (they are not probabilities but since the data has been transformed the scales might be in [0,1], see discussion at https://github.com/keras-team/keras/issues/108)
  # very good performance since a lower bound of MSE is given by sigma^2/batch_len=(0.25/sqrt(0.5))/501 where 1/sqrt(0.5) is due to data transformation ([-1,1] into [0,1])
  #  model %>% predict_proba(x_test)
  preds_scaled<-model %>% predict(x_test)
  
  ts.plot(preds_scaled,col="red",main="True (blue) vs. predicted (red) out-of-sample")
  lines(y_test,col="blue")
  
  predicted_keras_lstm<-preds_scaled*(max(data_mat[,1])-min(data_mat[,1]))+min(data_mat[,1])
  
  test.r <- y_test*(max(data_mat[,1])-min(data_mat[,1]))+min(data_mat[,1])
  # Calculating MSE
  MSE.keras_lstm <- mean((test.r - predicted_keras_lstm)^2)
  
  return(list(MSE.keras_lstm=MSE.keras_lstm,predicted_keras_lstm=predicted_keras_lstm))
  
}






keras_recurrent_lstm_predict<-function(data_obj,number_neurons,period_sharpe,epochs,anz_real)
{
  train_set<-data_obj$train_set
  test_set<-data_obj$test_set
  target_out<-data_obj$target_out
  target_in<-data_obj$target_in
  data_mat<-data_obj$data_mat
  
  target_train<-train_set[,1]
  explanatory_train<-as.matrix(train_set[,2:ncol(train_set)])
  target_test<-test_set[,1]
  explanatory_test<-as.matrix(test_set[,2:ncol(train_set)])
  
  # This is a particular formatting of the data for lstm-nets in keras (it differs from rrn package or mxnet)  
  y_train<-as.matrix(train_set[,1])
  #  plot(y_train)
  dim(y_train)
  x_train<-array(as.matrix(train_set[,2:ncol(train_set)]),dim=c(dim(as.matrix(train_set[,2:ncol(train_set)])),1))
  dim(x_train)
  
  y_test<-as.matrix(test_set[,1])
  dim(y_test)
  x_test<-array(as.matrix(test_set[,2:ncol(test_set)]),dim=c(dim(as.matrix(test_set[,2:ncol(test_set)])),1))
  dim(x_test)
  
  
  
  MSE_mat<-matrix(ncol=1,nrow=anz_real)
  colnames(MSE_mat)<-"Out sample MSE"
  
  pb <- txtProgressBar(min = 1, max = anz_real, style = 3)
  batch_size <- nrow(explanatory_train)
  sharpe_keras<-1:anz_real
  # One could try alternative set.seeds and/or larger anz_real
  #  use_session_with_seed(1, disable_gpu = FALSE, disable_parallel_cpu = FALSE)
  for (i in 1:anz_real)#i<-1
  {
    
    
    # Note that loss and mean-squared error are identical in our case (since we selected MSE as performance measure)
    #   In-sample (training) and out-of-sample (validation) MSEs
    #   Slight overfitting visible for epochs>100
    fit_keras_net<-estimate_keras_lstm_func(x_train,y_train,x_test,y_test,number_neurons,data_mat,batch_size,epochs)
    
    predicted_keras<-fit_keras_net$predicted_keras_lstm
    if (i==1)
    {
      predicted_mat<-predicted_keras
    } else
    {
      predicted_mat<-cbind(predicted_mat,predicted_keras)
    }
    MSE_mat[i,]<-fit_keras_net$MSE.keras_lstm
    # Go long or short depending on sign of forecast
    #   We do not need to lag the signal here since the forecast is based on (already) lagged data 
    
    setTxtProgressBar(pb, i)
  }
  close(pb)
  return(list(MSE_mat=MSE_mat,predicted_mat=predicted_mat))
} 




# Recurrent rnn package
estimate_rnn_func<-function(x_train_rnn,y_train_rnn,x_test_rnn,y_test_rnn,number_neurons,data_mat,batch_size,epochs,network_type)
{
  
  model <- trainr(Y = y_train_rnn,
                  X = x_train_rnn,
                  learningrate = 0.05,
                  hidden_dim = number_neurons,
                  numepochs = epochs,
                  network_type=network_type)
  
  
  # Predicted values
  Yp_train <- predictr(model, x_train_rnn)
  par(mfrow=c(1,1))
  # Plot predicted vs actual. Training set + testing set
  plot(as.vector(t(y_train_rnn)), col = "red", type = "l", main = "Actual vs predicted", ylab = "Y,Yp")
  lines(as.vector(t(Yp_train)), type = "l", col = "blue")
  legend("topright", c("Predicted", "Real"), col = c("blue","red"), lty = c(1,1), lwd = c(1,1))
  
  preds_scaled<-predictr(model, x_test_rnn)
  
  predicted_rnn<-preds_scaled*(max(data_mat[,1])-min(data_mat[,1]))+min(data_mat[,1])
  
  test.r <- y_test_rnn*(max(data_mat[,1])-min(data_mat[,1]))+min(data_mat[,1])
  # Calculating MSE
  MSE.rnn <- mean((test.r - predicted_rnn)^2)
  
  return(list(MSE.rnn=MSE.rnn,predicted_rnn=predicted_rnn))
  
}








rnn_recurrent_predict<-function(data_obj,number_neurons,period_sharpe,epochs,network_type)
{
  train_set<-data_obj$train_set
  test_set<-data_obj$test_set
  target_out<-data_obj$target_out
  target_in<-data_obj$target_in
  data_mat<-data_obj$data_mat
  
  target_train<-train_set[,1]
  explanatory_train<-as.matrix(train_set[,2:ncol(train_set)])
  target_test<-test_set[,1]
  explanatory_test<-as.matrix(test_set[,2:ncol(train_set)])
  
  # This is a particular formatting of the data for rnn recurrent (it differs from keras package or mxnet)  
  
  y_train_rnn<-as.matrix(train_set[,1])
  plot(y_train_rnn)
  dim(y_train_rnn)
  x_train_rnn<-array(as.matrix(train_set[,2:ncol(train_set)]),dim=c(dim(as.matrix(train_set[,2:ncol(train_set)]))[1],1,dim(as.matrix(train_set[,2:ncol(train_set)]))[2]))
  dim(x_train_rnn)
  
  y_test_rnn<-as.matrix(test_set[,1])
  dim(y_test_rnn)
  
  x_test_rnn<-array(as.matrix(test_set[,2:ncol(test_set)]),dim=c(dim(as.matrix(test_set[,2:ncol(test_set)]))[1],1,dim(as.matrix(test_set[,2:ncol(test_set)]))[2]))
  dim(x_test_rnn)
  
  
  
  MSE_mat<-matrix(ncol=1,nrow=anz_real)
  colnames(MSE_mat)<-"Out sample MSE"
  
  pb <- txtProgressBar(min = 1, max = anz_real, style = 3)
  batch_size <- nrow(explanatory_train)
  sharpe_keras<-1:anz_real
  set.seed(0)
  
  for (i in 1:anz_real)#i<-1
  {
    
    
    # Note that loss and mean-squared error are identical in our case (since we selected MSE as performance measure)
    #   In-sample (training) and out-of-sample (validation) MSEs
    #   Slight overfitting visible for epochs>100
    fit_rnn.obj<-estimate_rnn_func(x_train_rnn,y_train_rnn,x_test_rnn,y_test_rnn,
                                   number_neurons,data_mat,batch_size,epochs,network_type)
    
    predicted_rnn<-fit_rnn.obj$predicted_rnn
    if (i==1)
    {
      predicted_mat<-predicted_rnn
    } else
    {
      predicted_mat<-cbind(predicted_mat,predicted_rnn)
    }
    MSE_mat[i,]<-fit_rnn.obj$MSE.rnn
    # Go long or short depending on sign of forecast
    #   We do not need to lag the signal here since the forecast is based on (already) lagged data 
    
    setTxtProgressBar(pb, i)
  }
  close(pb)
  return(list(MSE_mat=MSE_mat,predicted_mat=predicted_mat))
} 






sharpe_func<-function(perf,period_len)
{
  return(as.double(sqrt(period_len)*mean(na.exclude(perf))/sqrt(var(na.exclude(perf)))))
}



read_main_func_monthly<-function(reload_data,in_out_sample_separator)#(path.dat,roll_day,subsetting,agg_samp,sampling,anf_day_char,end_day_char)
{
  # Reload data if reload_data==T or if no data was previously stored  
  if (reload_data)
  {
    end_date<-format(Sys.time(), "%Y-%m-%d")
    start_date<-'1990-01-01'
    # we first load INDPRO in order to have the full time index (if GDP is loaded first then the time index is based on quarters...)    
    ser_load<-Quandl('FRED/INDPRO',start_date=start_date,end_date=end_date,type='xts')
    mydata<-as.xts(na.omit(ser_load))
    dependent_var<-'FRED/INDPRO'
    ser_load<-Quandl(dependent_var,start_date=start_date,end_date=end_date,type='xts')
    mydata<-cbind(mydata,as.xts(na.omit(ser_load)))
    colnames(mydata)[2]<-dependent_var
    is.xts(mydata)
    # Alternative download: Real Gross Domestic Product, 3 Decimal
    
    explaining_data<-c('FRED/UNRATE','FRED/PAYEMS','FRED/INDPRO','CHRIS/CME_SP1')
    explaining_data<-c('FRED/UNRATE','FRED/PAYEMS','FRED/INDPRO','ISM/MAN_PMI')
    for (i in 1:length(explaining_data))#i<-5
    {
      # 1.  FRED/UNRATE   
      # 2.  FRED/PAYEMS   
      # 3.  FRED/INDPRO
      ser_load<-Quandl(explaining_data[i],start_date=start_date,end_date=end_date,type='xts')
      ser_load<-diff(log(ser_load))
      # We lag the series by one month since quandl-date is shifted by one month 
      #   Quandl does not refer to publication date but to period to which data refers
      mydata<-cbind(mydata,ser_load)
      colnames(mydata)[ncol(mydata)]<-explaining_data[i]
    }  
    # Skip first series: s&p in first column is redundant (was used for time index only...)
    mydata<-mydata[,-(1:2)]
    ser_load<-Quandl('CHRIS/CME_SP1',start_date=start_date,end_date=end_date,type='xts')
    ser_load<-diff(log(ser_load)[,"Open"])
    ser_load<-na.locf(ser_load)
    monthly_leg_ret_sp<-apply.monthly(ser_load,sum)
    if (length(monthly_leg_ret_sp)==nrow(mydata)+1)
      monthly_leg_ret_sp<-monthly_leg_ret_sp[-length(monthly_leg_ret_sp)]
    index(monthly_leg_ret_sp)<-index(mydata)
    mydata<-cbind(mydata,monthly_leg_ret_sp)
    is.xts(mydata)
    tail(mydata)
    head(mydata)
    colnames(mydata)<-c("Unrate","Payroll","Indpro","PMI","SP monthly returns")
    mydata<-na.exclude(mydata)
    print(paste(getwd(),"/Exercises/Dritte Woche/SP",sep=""))
    save(mydata, file = paste(getwd(),"/SP.Rdata",sep=""))
    
    #    write.table(mydata,file=paste(path.dat,"mixed",sep=""))
  } else
  {
    
    load(file = paste(getwd(),"/SP.Rdata",sep=""))    #YM,URO,SF,NQ,ES,DX,CD,AD  is.xts(tsData)
  }
  sp_data<-mydata
  lag_explanatory<-1
  
  # Target series in first column
  # Select explanatory series: PMI is 'suspect' (wrong sign in regression: cannot be analyzed in nn)
  exp_select<-c("SP monthly returns","Unrate","Payroll","Indpro","PMI")
  exp_select<-c("SP monthly returns","Unrate","Payroll","Indpro")
  
  data_mat<-na.exclude(cbind(sp_data[,"SP monthly returns"],lag(sp_data[,exp_select],k=lag_explanatory)))
  
  head(data_mat)
  
  
  
  target_in<-data_mat[paste("/",in_out_sample_separator,sep=""),1]
  tail(target_in)
  explanatory_in<-data_mat[paste("/",in_out_sample_separator,sep=""),2:ncol(data_mat)]
  tail(explanatory_in)
  
  target_out<-data_mat[paste(in_out_sample_separator,"/",sep=""),1]
  head(target_out)
  tail(target_out)
  explanatory_out<-data_mat[paste(in_out_sample_separator,"/",sep=""),2:ncol(data_mat)]
  head(target_out)
  tail(explanatory_out)
  
  # Scaling data for the NN
  maxs <- apply(data_mat, 2, max) 
  mins <- apply(data_mat, 2, min)
  # Transform data into [0,1]  
  scaled <- scale(data_mat, center = mins, scale = maxs - mins)
  
  apply(scaled,2,min)
  apply(scaled,2,max)
  #-----------------
  # 4.b
  # Train-test split
  train_set <- scaled[paste("/",in_out_sample_separator,sep=""),]
  test_set <- scaled[paste(in_out_sample_separator,"/",sep=""),]
  
  train_set<-as.matrix(train_set)
  test_set<-as.matrix(test_set)
  
  return(list(data_mat=data_mat,target_in=target_in,target_out=target_out,
              explanatory_in=explanatory_in,explanatory_out=explanatory_out,
              train_set=train_set,test_set=test_set))
}




#.####
# NN####
# Train/Predict various neural nets
# - Define first the data_obj with the funtion: data_function
#   x: data
#   lags: amount of lags (e.g. acf(x) -> significant = 6 -> lags=6)
#   start: Startdate (e.g. 2020-01-01)
#   end: Enddate (eg. 2020-07-31)
#   in_out_sep: Seperator (eg. 2020-07-01)
#   data_function will return a few data objects which are needed for
#   the computation of the MSE and Sharpe

# - use nn_nl_comb_sharpe_mse
#   define maxneurons/layers and the amount of realizations

## MSE Sharpe Function####
nn_nl_comb_sharpe_mse <- function(maxneuron=3, maxlayer=3, real=10, data_obj) {
  starttime=Sys.time()
  # Define Input Grid
  # needs input grid function
  combmat <- input_grid(maxneuron,maxlayer)
  
  
  # Naming the  grid with combinations
  ind <- rep(NA,dim(combmat)[1])
  for(k in 1:dim(combmat)[1])
  {
    x <- as.vector(combmat[k,])
    ind[k] <- toString(as.character(x[x!=0]))
  }
  
  # Define result matrix
  mati <- matrix(nrow=dim(combmat)[1], ncol=real*4, 0)
  mati <- as.data.frame(mati)
  rownames(mati) <- ind
  
  
  #creating , testing , neural net
  for( i in 1: dim(combmat)[1]) {
    pb <- txtProgressBar(min = 1, max = dim(combmat)[1], style = 3)
    
    x=as.vector(combmat[i,])
    x= x[x!=0]
    
    for(k in seq(1,real*4,4)) {
      
      net <- nn_estim(data_obj, nl_comb=x)
      
      mati[i, k:(k+3)] <- c(net$mse_nn, net$sharpe_nn)
    }
    
    cat("\014")
    print(paste("Elapsed Time: " ,Sys.time()-starttime))
    print(paste("Iteration: ", i, "of", dim(combmat)[1]))
    setTxtProgressBar(pb, i)
    
  }
  
  # close(pb)
  print(paste("Overall Time: " ,Sys.time()-starttime))
  return(mati)
}

## Data Function####
data_function <- function(x, lags, in_out_sep, start="", end="",autoassign=F) {
  # Define startpoints
  x <- x[paste(start,"::", end, sep="")]
  data_mat <- x
  
  # Create lagged data
  for (j in 1:lags)
    data_mat <- cbind(data_mat, lag(x, k=j))
  
  # Remove NA's
  data_mat <- na.exclude(data_mat)
  
  
  # Specify in- and out-of-sample episodes
  # Target in-sample (current data)
  target_in <- data_mat[paste("/",in_out_sep,sep=""),1]
  # Remove last value
  target_in <- target_in[1:(length(target_in)-1),1]
  
  # Target out of sample (current data)
  target_out <- data_mat[paste(in_out_sep,"/",sep=""),1]
  
  # Scaling data for the NN
  maxs <- apply(data_mat, 2, max)
  mins <- apply(data_mat, 2, min)
  # Transform data into [0,1]
  scaled <- scale(data_mat, center = mins, scale = maxs - mins)
  
  train_set_xts <- scaled[paste("/",in_out_sep,sep=""),]
  test_set_xts <- scaled[paste(in_out_sep,"/",sep=""),]
  # Train-test split
  train_set <- scaled[paste("/",in_out_sep,sep=""),]
  # Remove last value
  train_set <- train_set[1:(dim(train_set)[1]-1),]
  
  test_set <- scaled[paste(in_out_sep,"/",sep=""),]
  
  train_set <- as.matrix(train_set)
  test_set <- as.matrix(test_set)
  
  # Formula
  colnames(train_set) <- paste("lag",0:(ncol(train_set)-1),sep="")
  n <- colnames(train_set)
  f <- as.formula(paste("lag0 ~", paste(n[!n %in% "lag0"], collapse = " + ")))
  
  if(autoassign)
  {
    assign("data_mat",data_mat,.GlobalEnv)
    assign("target_in",target_in,.GlobalEnv)
    assign("target_out",target_out,.GlobalEnv)
    assign("train_set",train_set,.GlobalEnv)
    assign("test_set",test_set,.GlobalEnv)
    assign("train_set_xts",train_set_xts,.GlobalEnv)
    assign("test_set_xts",test_set_xts,.GlobalEnv)
    assign("f",f,.GlobalEnv)
  }
  
  
  return(list(data_mat=data_mat,
              target_in=target_in,
              target_out=target_out,
              train_set=train_set,
              test_set=test_set,
              f=f))
}




## Estimate Fun####
nn_estim <- function(data_obj, nl_comb) {
  
  # Prepare data
  train_set <- data_obj$train_set
  test_set <- data_obj$test_set
  data_mat <- data_obj$data_mat
  target_in <- data_obj$target_in
  target_out <- data_obj$target_out
  f <- as.formula(data_obj$f)
  
  
  # Train NeuralNet
  nn <- neuralnet(f, data=train_set, hidden=nl_comb, linear.output=T, stepmax = 1e+08)
  
  
  # In sample performance
  pred_in_scaled <- nn$net.result[[1]]
  # Scale back from interval [0,1] to original log-returns
  pred_in <- pred_in_scaled*(max(data_mat[,1])-min(data_mat[,1]))+min(data_mat[,1])
  # In-sample MSE
  train_rescaled <- train_set[,1]*(max(data_mat[,1])-min(data_mat[,1]))+min(data_mat[,1])
  mse_in <- mean((train_rescaled - pred_in)^2)
  
  # In-sample Sharpe
  perf_in <- (sign(pred_in))*target_in
  sharpe_in <- as.numeric(sqrt(365)*mean(perf_in)/sqrt(var(perf_in)))
  
  # Out-of-sample performance
  # Compute out-of-sample forecasts
  # pr.nn <- compute(nn, as.matrix(test_set[,2:ncol(test_set)]))
  pr.nn <- predict(nn, as.matrix(test_set[,2:ncol(test_set)]))
  
  predicted_scaled <- pr.nn
  # Results from NN are normalized (scaled)
  # Descaling for comparison
  pred_out <- predicted_scaled*(max(data_mat[,1])-min(data_mat[,1]))+min(data_mat[,1])
  test_rescaled <- test_set[,1]*(max(data_mat[,1])-min(data_mat[,1]))+min(data_mat[,1])
  # Calculating MSE
  mse_out <- mean((test_rescaled - pred_out)^2)
  
  # Out of sample Sharpe
  perf_out <- (sign(pred_out))*target_out
  sharpe_out <- sqrt(365)*mean(perf_out)/sqrt(var(perf_out))
  
  # Compare in-sample and out-of-sample
  mse_nn <- c(mse_in, mse_out)
  sharpe_nn <- c(sharpe_in, sharpe_out)
  
  return(list(mse_nn=mse_nn, pred_out=pred_out, pred_in=pred_in, sharpe_nn=sharpe_nn))
}

## Input Grid Function####
input_grid <- function(n=3, l=3) {
  anz <- n^(1:l)
  mat <- matrix(0, nrow=sum(anz), ncol=l)
  
  
  i_end <- cumsum(anz)
  i_start <- anz-1
  i_start <- i_end - i_start
  
  
  for(j in 0:(length(anz)-1)) {
    for (i in (1+j):l) {
      mat[i_start[i]:i_end[i], i-j] <- rep(1:n, rep(n^(j), n))
    }
  }
  return(as.data.frame(mat))
}



## RNN####
rnn_nl_comb_sharpe_mse <- function(maxneuron=3, maxlayer=3, real=10, data_obj, epochs=10, nn_type="rnn", learningrate=0.05) {
  starttime=Sys.time()
  # Define Input Grid
  # needs input grid function
  combmat <- input_grid(maxneuron,maxlayer)
  
  
  # Naming the  grid with combinations
  ind <- rep(NA,dim(combmat)[1])
  for(k in 1:dim(combmat)[1])
  {
    x <- as.vector(combmat[k,])
    ind[k] <- toString(as.character(x[x!=0]))
  }
  
  # Define result matrix
  mati <- matrix(nrow=dim(combmat)[1], ncol=real*4, 0)
  mati <- as.data.frame(mati)
  rownames(mati) <- ind
  
  
  #creating , testing , neural net
  for( i in 1: dim(combmat)[1]) {
    pb <- txtProgressBar(min = 1, max = dim(combmat)[1], style = 3)
    
    x=as.vector(combmat[i,])
    x= x[x!=0]
    
    for(k in seq(1,real*4,4)) {
      
      net <- rnn_estim(data_obj, nl_comb=x, epochs, nn_type, learningrate)
      
      
      mati[i, k:(k+3)] <- c(net$mse_nn, net$sharpe_nn)
    }
    
    cat("\014")
    print(paste("Elapsed Time: " ,Sys.time()-starttime))
    print(paste("Iteration: ", i, "of", dim(combmat)[1]))
    setTxtProgressBar(pb, i)
    
  }
  
  # close(pb)
  print(paste("Overall Time: " ,Sys.time()-starttime))
  return(mati)
}


rnn_estim <- function(data_obj, nl_comb, epochs, nn_type, learningrate) {
  
  # Prepare data
  train_set <- data_obj$train_set
  test_set <- data_obj$test_set
  target_out <- data_obj$target_out
  target_in <- data_obj$target_in
  data_mat <- data_obj$data_mat
  
  
  # This is a particular formatting of the data for rnn recurrent (it differs from keras package or mxnet)  
  
  y_train_rnn <- as.matrix(train_set[,1])
  x_train_rnn <- array(as.matrix(train_set[,2:ncol(train_set)]),dim=c(dim(as.matrix(train_set[,2:ncol(train_set)]))[1],1,dim(as.matrix(train_set[,2:ncol(train_set)]))[2]))
  
  y_test_rnn <- as.matrix(test_set[,1])
  x_test_rnn <- array(as.matrix(test_set[,2:ncol(test_set)]),dim=c(dim(as.matrix(test_set[,2:ncol(test_set)]))[1],1,dim(as.matrix(test_set[,2:ncol(test_set)]))[2]))
  
  batch_size <- nrow(train_set)
  
  # Train NeuralNet
  model <- trainr(Y = y_train_rnn,
                  X = x_train_rnn,
                  learningrate = learningrate,
                  hidden_dim = nl_comb,
                  numepochs = epochs,
                  nn_type=nn_type)
  
  # In sample performance
  pred_in_scaled <- predictr(model, x_train_rnn)
  # Scale back from interval [0,1] to original log-returns
  pred_in <- pred_in_scaled*(max(data_mat[,1])-min(data_mat[,1]))+min(data_mat[,1])
  # In-sample MSE
  train_rescaled <- train_set[,1]*(max(data_mat[,1])-min(data_mat[,1]))+min(data_mat[,1])
  mse_in <- mean((train_rescaled - pred_in)^2)
  
  # In-sample Sharpe
  perf_in <- (sign(pred_in))*target_in
  sharpe_in <- as.numeric(sqrt(365)*mean(perf_in)/sqrt(var(perf_in)))
  
  # Out-of-sample performance
  # Compute out-of-sample forecasts
  pr.nn <- predictr(model, x_test_rnn)
  
  predicted_scaled <- pr.nn
  # Results from NN are normalized (scaled)
  # Descaling for comparison
  pred_out <- predicted_scaled*(max(data_mat[,1])-min(data_mat[,1]))+min(data_mat[,1])
  test_rescaled <- test_set[,1]*(max(data_mat[,1])-min(data_mat[,1]))+min(data_mat[,1])
  # Calculating MSE
  mse_out <- mean((test_rescaled - pred_out)^2)
  
  # Out of sample Sharpe
  perf_out <- (sign(pred_out))*target_out
  sharpe_out <- sqrt(365)*mean(perf_out)/sqrt(var(perf_out))
  
  # Compare in-sample and out-of-sample
  mse_nn <- c(mse_in, mse_out)
  sharpe_nn <- c(sharpe_in, sharpe_out)
  
  return(list(mse_nn=mse_nn, pred_out=pred_out, pred_in=pred_in, sharpe_nn=sharpe_nn))
}
