setwd('~/python/Theanolearn/facecnn/lstm')
X=read.csv('strat1.csv')
plot(X$sp,type='l',xlab='days',ylab='revenue rate',ylim=c(-0.14,0.02))
lines(X$str,col=2,lty=2)
legend(10,-0.07,c('s$p index','strategy'),col=c(1,2),lty=c(1,2))

