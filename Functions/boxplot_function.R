var_boxplot <- function(var){
  
  #reshape data to long format 
  var_long <- var %>%
    pivot_longer(everything(), names_to = "variable", values_to = "value")
  
  #Create a boxplot using ggplot2
  ggplot(var_long, aes(x = variable, y = value)) + 
    geom_boxplot()
  
}