dv<- function(RT, Resp, color ="lightgray"){
  Resp<- as.data.frame(Resp)
  RT<- as.data.frame(RT)
  nresp<- nrow(Resp)
  nrt<- ncol(RT)
  nresp_cols<- ncol(Resp)
  if (nrt != nresp_cols) stop("RT and Resp must have the same number of columns. RT: ", nrt, ", Resp: ", nresp_cols)
  if (nresp == 0L) stop("Resp has no rows")
  if (nrt == 0L) stop("RT has no columns")
  loop <- 1:nrt
  p <- colSums(Resp)
  p1<- round(p/nresp, 2)

  old.par <- par(mfrow = c(1,1), mar = c(5.1, 4.1, 4.1, 2.1),oma =c(0,0,0,0))

  par(mfrow = c(3,4), oma =c(0.5,0.5,0.5,0.5), mar=c(2,2,3,2))

  #i <- 1
  
  for (i in loop){
    x<-RT[,i]
    hist(x,
         main = paste("RT Distr. for Item",i),col= color,
         breaks = 15)
    mtext(paste("Correct Proportion:", p1[i]), cex= 0.5, col=2)

  }
  par(old.par)
}

# Only run when RT and Resp exist in the workspace (e.g. interactive R).
# When sourced from Python, we only define dv(); Python calls dv(rt_df, resp_df, ...).
if (exists("RT") && exists("Resp")) {
  dv(RT, Resp)
}

