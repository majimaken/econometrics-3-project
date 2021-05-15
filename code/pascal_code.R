## FFN (1, 4)
## RNN (7)
## LSTM (9, 8, 2)
## GRU (8, 9)
rm(list = ls())
source("add/libraries.R")
source("add/Functions_RNN.R")




load("data/ETH_2021-05-05.rda")
head(ETH)
tail(ETH)

# Define log returns based on closing prices

logret <- diff(log(ETH$`ETH-USD.Close`))
logret <- na.omit(logret)
colnames(logret) <- "ETH Log Returns"
subi=logret["2020-10-01::2021-04-30"]

in_out_sample_separator <- "2020-10-01"



data_obj <- data_function(x=logret, lags=10, in_out_sep="2021-04-01", start="2020-10-01", end="2021-04-30")


# Prepare data with lags 1-7
########################################################fnn################################################################
head(data_obj$data_mat)



anz=10000

outtarget=data_obj$target_out



for(i in 1:anz)
{
  if (i ==1 )
  {
    net=estimate_nn (train_set=data_obj$train_set,number_neurons=c(1,4),data_mat=data_obj$data_mat,test_set=data_obj$test_set,f=data_obj$f)
    
    
    perfall=cumsum(sign(net$predicted_nn) *outtarget)
    
    
  }else
  {
    net=estimate_nn (train_set=data_obj$train_set,number_neurons=c(1,4),data_mat=data_obj$data_mat,test_set=data_obj$test_set,f=data_obj$f)
    perf=cumsum(sign(net$predicted_nn) *outtarget)
    perfall=cbind(perfall,perf)
    
    
    
  }
}

mean=apply(perfall,1,mean)

mean=reclass(mean,perfall)
tail(perfall)



perfnew=merge(mean,cumsum(outtarget),perfall)




##################################lstm################################################################

## LSTM (9, 8, 2)

anz=100

  
nl_comb= 2 
                                   
epochs=  30                                 
nn_type="lstm"                                      
learningrate=0.05




for(i in 1:anz)
{
  if (i ==1 )
  {

    
    perfall=cumsum(rnn_estim(data_obj, nl_comb, epochs, nn_type, learningrate)$perf_out)
    
  }else
  {
    perf=cumsum(rnn_estim(data_obj, nl_comb, epochs, nn_type, learningrate)$perf_out)
    perfall=cbind(perfall,perf)
    cat("\014")
  }
}

outtarget=data_obj$target_out

mean=apply(perfall,1,mean)

mean=reclass(mean,outtarget)
perfall=reclass(perfall,outtarget)



perfnew_lstm=merge(mean,cumsum(outtarget),perfall)



save(perfnew_lstm, file="C:/Users/buehl/OneDrive/Dokumente/ZHAW/BSc Wirtschaftsingenieur/SEM8/Oeko3/econometrics-3-project/data/perfnew_lstm.rda")
#################################RNN################################################################



anz=100


nl_comb= c(10,9)

epochs=10                                     
nn_type="rnn"      ## RNN (7)                                
learningrate=0.05




for(i in 1:anz)
{
  if (i ==1 )
  {
    
    
    perfall=cumsum(rnn_estim(data_obj, nl_comb, epochs, nn_type, learningrate)$perf_out)
    
  }else
  {
    perf=cumsum(rnn_estim(data_obj, nl_comb, epochs, nn_type, learningrate)$perf_out)
    perfall=cbind(perfall,perf)
    cat("\014")
  }
}

outtarget=data_obj$target_out

mean=apply(perfall,1,mean)

mean=reclass(mean,outtarget)
perfall=reclass(perfall,outtarget)



perfnew_rnn=merge(mean,cumsum(outtarget),perfall)


save(perfnew_rnn, file="C:/Users/buehl/OneDrive/Dokumente/ZHAW/BSc Wirtschaftsingenieur/SEM8/Oeko3/econometrics-3-project/data/perfnew_rnn.rda")



#################################gru################################################################



anz=100

## GRU (8, 9)
nl_comb= c(6, 10)

epochs=30                                     
nn_type="gru"                                      
learningrate=0.05




for(i in 1:anz)
{
  if (i ==1 )
  {
    
    
    perfall=cumsum(rnn_estim(data_obj, nl_comb, epochs, nn_type, learningrate)$perf_out)
    
  }else
  {
    perf=cumsum(rnn_estim(data_obj, nl_comb, epochs, nn_type, learningrate)$perf_out)
    perfall=cbind(perfall,perf)
    cat("\014")
  }
}

outtarget=data_obj$target_out

mean=apply(perfall,1,mean)

mean=reclass(mean,outtarget)
perfall=reclass(perfall,outtarget)



perfnew_gru=merge(mean,cumsum(outtarget),perfall)



save(perfnew_gru, file="C:/Users/buehl/OneDrive/Dokumente/ZHAW/BSc Wirtschaftsingenieur/SEM8/Oeko3/econometrics-3-project/data/perfnew_gru.rda")


######################plots #############################################

par(mfrow=c(2,2))

##nn
anz=10000
plot(perfnew,col=c("red","black",rep("grey",anz)),lwd=c(7,7,rep(1,anz)),main="Out of sample Perfomance Feed forward net")

name=c("Buy and Hold","Mean of Nets")
addLegend("topleft", 
          legend.names=name,
          col=c("black","red"),
          lty=rep(1,1),
          lwd=rep(2,2),
          ncol=1,
          bg="white")

##lstm
anz=100
plot(perfnew_lstm,col=c("blue","black",rep("grey",anz)),lwd=c(7,7,rep(1,anz)),main="Out of sample Perfomance LSTM")

name=c("Buy and Hold","Mean of Nets")
addLegend("topleft", 
          legend.names=name,
          col=c("black","blue"),
          lty=rep(1,1),
          lwd=rep(2,2),
          ncol=1,
          bg="white")
##rnn
anz=100
plot(perfnew_rnn,col=c("orange","black",rep("grey",anz)),lwd=c(7,7,rep(1,anz)),main="Out of sample Perfomance rnn")

name=c("Buy and Hold","Mean of Nets")
addLegend("topleft", 
          legend.names=name,
          col=c("black","orange"),
          lty=rep(1,1),
          lwd=rep(2,2),
          ncol=1,
          bg="white")


##gru
anz=100
plot(perfnew_gru,col=c("green","black",rep("grey",anz)),lwd=c(7,7,rep(1,anz)),main="Out of sample Perfomance GRU")

name=c("Buy and Hold","Mean of Nets")
addLegend("topleft", 
          legend.names=name,
          col=c("black","green"),
          lty=rep(1,1),
          lwd=rep(2,2),
          ncol=1,
          bg="white")



#al together
name=c("Buy and Hold","feed forward","rnn","lstm","gru")
colors=c("black","red","blue","orange","green")

plot(cbind(perfnew_rnn[,2],perfnew[,1],perfnew_rnn[,1],perfnew_lstm[,1],perfnew_gru[,1]),main="Performance comparison",col=colors)

addLegend("topleft", 
          legend.names=name,
          col=colors,
          lty=rep(1,1),
          lwd=rep(2,2),
          ncol=1,
          bg="white")

