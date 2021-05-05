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
colnames(logret) <- "ETH Logarithmic Returns"
head(logret)
tail(logret)

par(mfrow = c(3,1))
ts.plot(logret)
acfPlot(logret)
pacfPlot(logret)

# Define in-sample (6m) and out-of-sample (1m)

len_in_sample <- 6
len_out_of_sample <- 1

in_out_sample_separator <- "2016-02-01"

# Prepare data with lags 1-7

data_obj <- data_function(logret, c(1:7), in_out_sample_separator)
head(data_obj$data_mat)

# Fit neural net 
network_type <- "rnn"
neurons <- c(7,7,7)
anz_real <- 5
period_sharpe <- 365
epochs <- 40

RNN_predict_obj <- rnn_recurrent_predict(data_obj,neurons, period_sharpe, epochs, network_type)

