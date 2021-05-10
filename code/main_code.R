# Preparation and data prep
rm(list = ls())
source("add/libraries.R")
source("add/Functions_RNN.R")


# Data loaded on 2021-05-05 from quandl
# getSymbols("ETH-USD") 
# ETH <- na.omit(`ETH-USD`)
# save(ETH, file = "data/ETH_2021-05-05.rda")

load("data/ETH_2021-05-05.rda")
head(ETH)
tail(ETH)

# Define log returns based on closing prices

logret <- diff(log(ETH$`ETH-USD.Close`))
logret <- na.omit(logret)
colnames(logret) <- "ETH Log Returns"
head(logret)
tail(logret)

# Data exploration and save plot
# png(file= "images/eth_exploration.png", width=550, height=550)
par(mfrow = c(3,1))
plot(ETH$`ETH-USD.Close`, col = 1, main = "ETH/USD", lwd = 0.7)
plot(log(ETH$`ETH-USD.Close`), col = 1, main = "Logarithmic ETH/USD", lwd = 0.7)
plot(logret, col = 1, main = "Logarithmic Returns ETH/USD", lwd = 0.7)
# dev.off()

# Check dependency structure and save plot
# png(file= "images/dependency.png", width=750, height=420)
par(mfrow = c(1,2))
acf(logret, main = "ACF")
pacf(logret, main = "PACF")
# dev.off()

# Find best ARIMA-model using auto.arima function from forecast package
fit <- auto.arima(logret, ic = "bic")
plot(forecast(fit))
tsdiag(fit) # looks tip top apart from vola clustering 

# Define in-sample (6m) and out-of-sample (1m)

len_in_sample <- 6
len_out_of_sample <- 1

in_out_sample_separator <- "2016-02-01"

# Prepare data with lags 1-7

data_obj <- data_function(logret, c(1:7), in_out_sample_separator)
head(data_obj$data_mat)

# Define parameters for fitting RNN
network_type <- "rnn"
neurons <- c(14,14,14)
anz_real <- 5
period_sharpe <- 365
epochs <- 40

# Fit RNN using rnn package #####
# network_type <- "rnn"
# neurons <- c(14,14,14)
# anz_real <- 5
# period_sharpe <- 365
# epochs <- 40

RNN_predict_obj <- rnn_recurrent_predict(data_obj,neurons, period_sharpe, epochs, network_type)
RNN_predict_obj$MSE_mat

# Fit RNN using keras package #####
# neurons <- c(14,14,14)
# anz_real<- 5
# period_sharpe<-365
# epochs<- 40

keras_recurrent_lstm_predict.obj<-keras_recurrent_lstm_predict(data_obj,neurons,period_sharpe,epochs,anz_real)
keras_recurrent_lstm_predict.obj$MSE_mat


