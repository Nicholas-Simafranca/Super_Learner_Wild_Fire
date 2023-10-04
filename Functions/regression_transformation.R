var_tran <- function(dates, var) {
  # create lags (days before/after fire ignition)
  diff <- dates - firedate
  lag <- as.numeric(diff)
  
  # transform variable of interest
  X <- matrix(0, nrow(var), 2)
  for (i in 1:nrow(var)) {
    original_variable <- as.numeric(var[i,])  # Convert to numeric vector
    X[i, ] <- lm(original_variable ~ lag)$coef
  }
  
  return(X)
}