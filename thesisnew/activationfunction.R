par(mfrow=c(2,2))
x=seq(from=-5,to=5,by=0.1)
plot(x,pmax(rep(0,length(x)),x), type="l")