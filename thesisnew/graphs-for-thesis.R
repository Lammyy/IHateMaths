q=0.5
p=0.5
ADVD=function(x){-q*log(x)-p*log(1-x)}
ADVR=function(x){-q*log(x/(1+x))+p*log(x+1)}
ADVT=function(x){-q*x+p*exp(x)}
KLD=function(x){-q*log(x/(1-x))+p*x/(1-x)}
KLR=function(x){-q*log(x)+p*x}
KLT=function(x){q*log((exp(x)+1)/exp(x))+p*(log(exp(x)+1))}
x=seq(0.01,0.99,by=0.001)
x1=seq(exp(-3.5),exp(3.5),by=0.001)
x2=seq(-3.5,3.5,by=0.001)
y=seq(0.1,1.6,by=0.01)
zer=rep(0,length(y))
one=rep(1,length(y))
par(mfrow=c(1,1))
plot(x,ADVD(x),type='l',xlim=c(-3,3),ylim=c(0.5,1.5),xlab="D(u)",ylab="Loss Function Value")
lines(zer,y,type='l',lty=2)
lines(one,y,type='l',lty=2)
plot(x1,ADVR(x1),type='l',xlim=c(-3,3),ylim=c(0.5,1.5),xlab="r(u)",ylab="Loss Function Value")
lines(zer,y,type='l',lty=2)
plot(x2,ADVT(x2),type='l',xlim=c(-3,3),ylim=c(0.5,1.5),xlab="T(u)",ylab="Loss Function Value")
plot(x2,KLT(x2),type='l',xlim=c(-3,3),ylim=c(0.5,1.5),xlab="T(u)",ylab="Loss Function Value")
plot(x1,KLR(x1),type='l',xlim=c(-3,3),ylim=c(0.5,1.5),xlab="r(u)",ylab="Loss Function Value")
lines(zer,y,type='l',lty=2)
plot(x,KLD(x),type='l',xlim=c(-3,3),ylim=c(0.5,1.5),xlab="D(u)",ylab="Loss Function Value")
lines(zer,y,type='l',lty=2)
lines(one,y,type='l',lty=2)