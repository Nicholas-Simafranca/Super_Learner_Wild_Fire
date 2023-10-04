map.heatmap <- function (lon, lat, data, 
                         xlim=NULL, ylim=NULL, zlim=NULL,
                         mainTitle="", legendTitle="") {
  
  library(ggplot2)  
  
  # Combine the data into a dataframe
  dfMap           <- as.data.frame(cbind(lon, lat, data)) 
  #combine vectors column-wise
  colnames(dfMap) <- c("lon", "lat", "Value")
  
  # Set limits for x, y, z if not specified by the user
  if (is.null(xlim)) { xlim <- range( lon,na.rm=TRUE) }
  if (is.null(ylim)) { ylim <- range( lat,na.rm=TRUE) }
  if (is.null(zlim)) { zlim <- range(data,na.rm=TRUE) }
  
  # Create the plot
  p <- ggplot(dfMap, aes(x=lon, y=lat, fill=Value)) + theme_bw()
  p <- p + geom_tile()
  p <- p + labs(title=paste(mainTitle,"\n",sep=""), x="lon", y="lat")
  p <- p + theme(plot.title = element_text(size = rel(1.5))) 
  p <- p + coord_fixed(ratio=1.1, xlim=xlim, ylim=ylim)
  p <- p + scale_fill_viridis_c(limits=zlim,
                                name=legendTitle) 
  return(p)
  }
