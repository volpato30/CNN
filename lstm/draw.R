setwd('~/python/Theanolearn/facecnn/lstm')
X=read.csv('strat1.csv')
plot(X$sp,type='l',xlab='ticks',ylab='revenue rate')
lines(X$str,col=2,lty=2)
legend(250,-0.004,c('s$p index','strategy'),col=c(1,2),lty=c(1,2))

