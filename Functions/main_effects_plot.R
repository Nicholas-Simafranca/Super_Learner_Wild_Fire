#install.packages("ale")
#install.packages("randomForest")

##############################################################
# FUNCTION TO MAKE MAIN EFFECT PLOTS
# x are the orginal covariates, like NDVI, in a matrix
# y are the fitted values (i.e., Kriging predictions) on the test set
# cols are the column indices of the 4-5 most important predictors
##############################################################

main_effects <- function(x,y,cols=NULL,xlim=NULL,ylim=NULL,p_test=0.5){
  
  library(randomForest)
  library(ale)
  if(is.null(cols)){cols = 1:ncol(x)}
  if(is.null(xlim)){xlim = range(as.vector(x))}
  if(is.null(ylim)){ylim = range(y)}
  
  test        <- runif(length(y))<p_test
  data_test   <- data.frame(y=y[test], x=x[test,])
  data_train  <- data.frame(y=y[!test], x=x[!test,])
  fit         <- randomForest(y~.,data=data_train)
  out         <- ale(data_test,fit)
  
  plot(NA,xlim=xlim,ylim=ylim,xlab="Covariate value",ylab="Main effect")
  for(j in 1:length(cols)){
    outj <- out[[cols[j]]]$data
    lines(outj$ale_x,outj$ale_y,lwd=2,col=j)
  }
  legend("bottomright",colnames(fake_x)[cols],col=1:length(cols),lwd=2,bty="n")
}
