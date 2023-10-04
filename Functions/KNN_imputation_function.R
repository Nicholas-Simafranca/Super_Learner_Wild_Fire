knn <- function(long,lat,X,k=10){
  miss <- which(is.na(X))
  full <- X
  for(i in miss){
    d       <- sqrt((long[i] - long)^2 + (lat[i]  - lat)^2)
    d[miss] <- Inf
    neigh   <- order(d)[1:k]
    full[i] <- mean(X[neigh])
  }
  return(full)
}