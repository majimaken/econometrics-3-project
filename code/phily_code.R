rm(list = ls())
source("add/libraries.R")
source("add/Functions_RNN.R")


# Etherium
load("data/ETH_2021-05-05.rda")
head(ETH)
tail(ETH)

ETH <- ETH["::2021-04-30"]

# Define log returns based on closing prices
logret <- diff(log(ETH$`ETH-USD.Close`))
logret <- na.omit(logret)
colnames(logret) <- "ETH_LR"
head(logret)
tail(logret)

# Plots
load("data/ETH_2021-05-05.rda")
ETH <- ETH["::2021-04-30"]
par(mfrow = c(3,1))
plot(ETH$`ETH-USD.Close`, col = 1, main = "ETH/USD", lwd = 0.7)
plot(log(ETH$`ETH-USD.Close`), col = 1, main = "Logarithmic ETH/USD", lwd = 0.7)
plot(logret, col = 1, main = "Logarithmic Returns ETH/USD", lwd = 0.7)


# ACF
par(mfrow = c(1,2))
acf(logret, main = "ACF")
pacf(logret, main = "PACF")


# subset
subi <- logret["2020-10-01::2021-04-30"]
par(mfrow = c(1,2))
acf(subi, main = "ACF")
pacf(subi, main = "PACF")
# Sicher bis lag = 10


head(logret)
data_obj <- data_function(x=logret, lags=10, in_out_sep="2021-04-01", start="2020-10-01", end="2021-04-30")
# save(data_obj, file="data/data_obj.rda")
c(as.character(time(head(data_obj$target_in, 1))), as.character(time(tail(data_obj$target_in, 1))), as.character(length(data_obj$target_in)))
c(as.character(time(head(data_obj$target_out, 1))), as.character(time(tail(data_obj$target_out, 1))), as.character(length(data_obj$target_out)))
fiti <- nn_nl_comb_sharpe_mse(maxneuron=10, maxlayer=3, real=100, data_obj)
# 10.27h
# optim_ffn <- fiti
# save(optim_ffn, file = "data/optim_ffn.rda")


fiti_rnn <- rnn_nl_comb_sharpe_mse(maxneuron=10,
                                   maxlayer=3,
                                   real=10,
                                   data_obj=data_obj,
                                   epochs=10,
                                   nn_type="rnn")
# 1.345h
# optim_rnn <- fiti_rnn
# save(optim_rnn, file = "data/optim_rnn.rda")

fiti_lstm <- rnn_nl_comb_sharpe_mse(maxneuron=10,
                                   maxlayer=3,
                                   real=10,
                                   data_obj=data_obj,
                                   epochs=10,
                                   nn_type="lstm")
# 1.344h
# optim_lstm <- fiti_lstm
# save(optim_lstm, file = "data/optim_lstm.rda")

fiti_gru <- rnn_nl_comb_sharpe_mse(maxneuron=10,
                                    maxlayer=3,
                                    real=10,
                                    data_obj=data_obj,
                                    epochs=10,
                                    nn_type="gru")
# 1.252h
# optim_gru <- fiti_gru
# save(optim_gru, file = "data/optim_gru.rda")

# MSE-IN####
par(mfrow=c(2,2))
hist(mean_ffn$mse_in)
hist(mean_rnn$mse_in)
hist(mean_lstm$mse_in)
hist(mean_gru$mse_in)

# MSE-OUT####
par(mfrow=c(2,2))
hist(mean_ffn$mse_out)
hist(mean_rnn$mse_out)
hist(mean_lstm$mse_out)
hist(mean_gru$mse_out)

# Sharpe-IN####
par(mfrow=c(2,2))
hist(mean_ffn$sharpe_in)
hist(mean_rnn$sharpe_in)
hist(mean_lstm$sharpe_in)
hist(mean_gru$sharpe_in)

# Sharpe-OUT####
par(mfrow=c(2,2))
hist(mean_ffn$sharpe_out)
hist(mean_rnn$sharpe_out)
hist(mean_lstm$sharpe_out)
hist(mean_gru$sharpe_out)



# Analysis####
head(optim_ffn)
tail(optim_ffn)
optim_ffn

# Calculate the means
meaner <- function(dat, real) {
  mse_in <- apply(X=dat[, seq(1, real, 4)], MARGIN=1, FUN=mean)
  mse_out <- apply(X=dat[, seq(2, real, 4)], MARGIN=1, FUN=mean)
  sharpe_in <- apply(X=dat[, seq(3, real, 4)], MARGIN=1, FUN=mean)
  sharpe_out <- apply(X=dat[, seq(4, real, 4)], MARGIN=1, FUN=mean)
  
  return(list(mse_in=mse_in, mse_out=mse_out, sharpe_in=sharpe_in, sharpe_out=sharpe_out))
}

mean_ffn <- meaner(optim_ffn, 100)
mean_rnn <- meaner(optim_rnn, 10)
mean_lstm <- meaner(optim_lstm, 10)
mean_gru <- meaner(optim_gru, 10)

save(mean_ffn, file="data/mean_ffn.rda")
save(mean_rnn, file="data/mean_rnn.rda")
save(mean_lstm, file="data/mean_lstm.rda")
save(mean_gru, file="data/mean_gru.rda")

# MSE-Plots ALL####
par_default <- par(no.readonly = TRUE)
par(mfrow=c(2,1), mar=c(4,5,3,2))
plot(mean_ffn$mse_in,
     type="l",
     ylab="MSE In-Sample",
     xlab="",
     xaxt="n",
     main="Neuron-Layer Combination: ETH/USD",
     frame.plot = FALSE,
     ylim=c(min(mean_ffn$mse_in, mean_rnn$mse_in, mean_lstm$mse_in, mean_gru$mse_in),
            max(mean_ffn$mse_in, mean_rnn$mse_in, mean_lstm$mse_in, mean_gru$mse_in)))
lines(mean_rnn$mse_in, col=2)
lines(mean_lstm$mse_in, col=3)
lines(mean_gru$mse_in, col=4)
axis(1, at=c(1, 110, 222, 444, 666, 888, 1110),
     labels=c("(1)", "(10,10)", "(2,2,2)","(4,4,4)", "(6,6,6)", "(8,8,8)", "(10,10,10)"))
rect(xleft=1,
     xright=10,
     ybottom=par('usr')[3],
     ytop=par('usr')[4],
     col="#FF00001A")

rect(xleft=11,
     xright=110,
     ybottom=par('usr')[3],
     ytop=par('usr')[4],
     col="#00FFFF1A")

rect(xleft=111,
     xright=1110,
     ybottom=par('usr')[3],
     ytop=par('usr')[4],
     col="#8000FF1A")


plot(mean_ffn$mse_out,
     type="l",
     ylab="MSE Out-of-Sample",
     xlab="",
     xaxt="n",
     frame.plot = FALSE,
     ylim=c(min(mean_ffn$mse_out, mean_rnn$mse_out, mean_lstm$mse_out, mean_gru$mse_out),
            max(mean_ffn$mse_out, mean_rnn$mse_out, mean_lstm$mse_out, mean_gru$mse_out)))
lines(mean_rnn$mse_out, col=2)
lines(mean_lstm$mse_out, col=3)
lines(mean_gru$mse_out, col=4)
axis(1, at=c(1, 110, 222, 444, 666, 888, 1110),
     labels=c("(1)", "(10,10)", "(2,2,2)","(4,4,4)", "(6,6,6)", "(8,8,8)", "(10,10,10)"))
rect(xleft=1,
     xright=10,
     ybottom=par('usr')[3],
     ytop=par('usr')[4],
     col="#FF00001A")

rect(xleft=11,
     xright=110,
     ybottom=par('usr')[3],
     ytop=par('usr')[4],
     col="#00FFFF1A")

rect(xleft=111,
     xright=1110,
     ybottom=par('usr')[3],
     ytop=par('usr')[4],
     col="#8000FF1A")
par(par_default)

legend("right", legend = c("FFN", "RNN", "LSTM", "GRU"), lty=1, pt.cex=2, cex=0.8, bty='n',
       col = 1:4, horiz=TRUE)
legend("left", legend=c("1 Layer", '2 Layers', '3 Layers'), pch=15, pt.cex=2, cex=0.8, bty='n',
       col = c('#FF00001A', '#00FFFF1A', '#8000FF1A'), horiz=TRUE)
legend("left", legend=c("1 Layer", '2 Layers', '3 Layers'), pch=15, pt.cex=2, cex=0.8, bty='n',
       col = c('#FF00001A', '#00FFFF1A', '#8000FF1A'), horiz=TRUE)



# MSE-Plots ohne FFN####
par_default <- par(no.readonly = TRUE)
par(mfrow=c(2,1), mar=c(4,5,3,2))
plot(mean_rnn$mse_in,
     type="l",
     ylab="MSE In-Sample",
     xlab="",
     main="Neuron-Layer Combination: ETH/USD",
     col=2,
     xaxt="n",
     frame.plot = FALSE,
     ylim=c(min(mean_rnn$mse_in, mean_lstm$mse_in, mean_gru$mse_in),
            max(mean_rnn$mse_in, mean_lstm$mse_in, mean_gru$mse_in)))
lines(mean_lstm$mse_in, col=3)
lines(mean_gru$mse_in, col=4)
axis(1, at=c(1, 110, 222, 444, 666, 888, 1110),
     labels=c("(1)", "(10,10)", "(2,2,2)","(4,4,4)", "(6,6,6)", "(8,8,8)", "(10,10,10)"))
rect(xleft=1,
     xright=10,
     ybottom=par('usr')[3],
     ytop=par('usr')[4],
     col="#FF00001A")

rect(xleft=11,
     xright=110,
     ybottom=par('usr')[3],
     ytop=par('usr')[4],
     col="#00FFFF1A")

rect(xleft=111,
     xright=1110,
     ybottom=par('usr')[3],
     ytop=par('usr')[4],
     col="#8000FF1A")

plot(mean_rnn$mse_out,
     type="l",
     ylab="MSE Out-of-Sample",
     xlab="",
     col=2,
     xaxt="n",
     frame.plot = FALSE,
     ylim=c(min(mean_rnn$mse_out, mean_lstm$mse_out, mean_gru$mse_out),
            max(mean_rnn$mse_out, mean_lstm$mse_out, mean_gru$mse_out)))
lines(mean_lstm$mse_out, col=3)
lines(mean_gru$mse_out, col=4)
axis(1, at=c(1, 110, 222, 444, 666, 888, 1110),
     labels=c("(1)", "(10,10)", "(2,2,2)","(4,4,4)", "(6,6,6)", "(8,8,8)", "(10,10,10)"))
rect(xleft=1,
     xright=10,
     ybottom=par('usr')[3],
     ytop=par('usr')[4],
     col="#FF00001A")

rect(xleft=11,
     xright=110,
     ybottom=par('usr')[3],
     ytop=par('usr')[4],
     col="#00FFFF1A")

rect(xleft=111,
     xright=1110,
     ybottom=par('usr')[3],
     ytop=par('usr')[4],
     col="#8000FF1A")

par(par_default)
legend("right", legend = c("RNN", "LSTM", "GRU"), lty=1, pt.cex=2, cex=0.8, bty='n',
       col = 2:4, horiz=TRUE)
legend("left", legend=c("1 Layer", '2 Layers', '3 Layers'), pch=15, pt.cex=2, cex=0.8, bty='n',
       col = c('#FF00001A', '#00FFFF1A', '#8000FF1A'), horiz=TRUE)
legend("left", legend=c("1 Layer", '2 Layers', '3 Layers'), pch=15, pt.cex=2, cex=0.8, bty='n',
       col = c('#FF00001A', '#00FFFF1A', '#8000FF1A'), horiz=TRUE)






# Sharpe-Plots####
perf_in <- as.numeric(data_obj$target_in)
perf_out <- as.numeric(data_obj$target_out)
sharpe_bnh_in <- as.numeric(sqrt(365)*mean(perf_in)/sqrt(var(perf_in)))
sharpe_bnh_out <- as.numeric(sqrt(365)*mean(perf_out)/sqrt(var(perf_out)))

# mean_ffn
# mean_rnn
# mean_lstm
# mean_gru

par_default <- par(no.readonly = TRUE)
par(mfrow=c(2,1), mar=c(4,5,3,2))
plot(mean_ffn$sharpe_in,
     type="l",
     ylab="Sharpe In-Sample",
     xlab="",
     main="Neuron-Layer Combination: ETH/USD",
     xaxt="n",
     frame.plot = FALSE,
     ylim=c(min(mean_ffn$sharpe_in, mean_rnn$sharpe_in, mean_lstm$sharpe_in, mean_gru$sharpe_in),
            max(mean_ffn$sharpe_in, mean_rnn$sharpe_in, mean_lstm$sharpe_in, mean_gru$sharpe_in)))
lines(mean_rnn$sharpe_in, col=2)
lines(mean_lstm$sharpe_in, col=3)
lines(mean_gru$sharpe_in, col=4)
abline(h=sharpe_bnh_in, lty=3, col=8)
axis(1, at=c(1, 110, 222, 444, 666, 888, 1110),
     labels=c("(1)", "(10,10)", "(2,2,2)","(4,4,4)", "(6,6,6)", "(8,8,8)", "(10,10,10)"))
rect(xleft=1,
     xright=10,
     ybottom=par('usr')[3],
     ytop=par('usr')[4],
     col="#FF00001A")

rect(xleft=11,
     xright=110,
     ybottom=par('usr')[3],
     ytop=par('usr')[4],
     col="#00FFFF1A")

rect(xleft=111,
     xright=1110,
     ybottom=par('usr')[3],
     ytop=par('usr')[4],
     col="#8000FF1A")

plot(mean_ffn$sharpe_out,
     type="l",
     ylab="Sharpe Out-of-Sample",
     xlab="",
     xaxt="n",
     frame.plot = FALSE,
     ylim=c(min(mean_ffn$sharpe_out, mean_rnn$sharpe_out, mean_lstm$sharpe_out, mean_gru$sharpe_out),
            max(mean_ffn$sharpe_out, mean_rnn$sharpe_out, mean_lstm$sharpe_out, mean_gru$sharpe_out)))
lines(mean_rnn$sharpe_out, col=2)
lines(mean_lstm$sharpe_out, col=3)
lines(mean_gru$sharpe_out, col=4)
abline(h=sharpe_bnh_out, lty=3, col=8)
axis(1, at=c(1, 110, 222, 444, 666, 888, 1110),
     labels=c("(1)", "(10,10)", "(2,2,2)","(4,4,4)", "(6,6,6)", "(8,8,8)", "(10,10,10)"))
rect(xleft=1,
     xright=10,
     ybottom=par('usr')[3],
     ytop=par('usr')[4],
     col="#FF00001A")

rect(xleft=11,
     xright=110,
     ybottom=par('usr')[3],
     ytop=par('usr')[4],
     col="#00FFFF1A")

rect(xleft=111,
     xright=1110,
     ybottom=par('usr')[3],
     ytop=par('usr')[4],
     col="#8000FF1A")

par(par_default)
legend("right", legend = c("FFN", "RNN", "LSTM", "GRU", "BnH-Sharpe"), lty=c(1,1,1,1,3), pt.cex=2, cex=0.8, bty='n',
       col = c(1,2,3,4,8), horiz=TRUE)
legend("left", legend=c("1 Layer", '2 Layers', '3 Layers'), pch=15, pt.cex=2, cex=0.8, bty='n',
       col = c('#FF00001A', '#00FFFF1A', '#8000FF1A'), horiz=TRUE)
legend("left", legend=c("1 Layer", '2 Layers', '3 Layers'), pch=15, pt.cex=2, cex=0.8, bty='n',
       col = c('#FF00001A', '#00FFFF1A', '#8000FF1A'), horiz=TRUE)