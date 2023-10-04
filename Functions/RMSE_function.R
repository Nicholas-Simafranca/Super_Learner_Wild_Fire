#RMSE function 
RMSE <- function(actual, predicted){
  residuals <- actual - predicted
  squared_residuals <- residuals ^2
  mean_squared <- sum(squared_residuals) / length(actual)
  RMSE <- sqrt(mean_squared)
  return(RMSE)
}