#' prepftr
#'
#' prepftr prepares the features to be used in supervised and unsupervised machine
#' learning models
#'
#' @param SecMin.obj objects from class \code{SecMin}
#' @param RT A required \eqn{N × J} \code{matrix} containing the response
#' times of \eqn{N} individuals to \eqn{J} items. Zero and missing values need
#' to be coded as \code{NA}.
#' @param Resp A required \eqn{N × J} \code{matrix} or \code{data.frame}
#' containing the item responses of \eqn{N} individuals to \eqn{J} items. The
#' correct responses are marked by 1 and incorrect responses by 0. Missing values
#' need to be coded as \code{NA}.
#' @param Flgd A required vector containing whether the individual is marked as
#' cheating or not. Non-cheaters need to be marked by 1 and cheaters need to be
#' marked by 0.
#' @param Info A non-required \eqn{N × I} \code{matrix} or \code{data.frame}
#' containing the background information of \eqn{N} individuals to \eqn{I} variables.
#'
#' @import caret
#' @importFrom mirt mirt coef fscores
#' @import stats
#'
#' @return \code{prepftr} returns the name and variable number of highly correlated
#' variables and saves two datasets, \code{full.features} and \code{selected.features}.
#' \code{full.features} include LZ values, Ht values,
#' Lzstart values, theta values from IRT analysis, person speed estimates, item
#' responses, and RTs. \code{selected.features} are \code{full.features} excluding
#' the highly correlated variables.
#'
#' @export



prepftr <- function(SecMin.obj, RT, Resp, Flgd, Info = NULL) {

  ############################# IRT model fitting  #############################

  fit.A <- mirt(Resp,1,itemtype='Rasch')
  #fit.A
  fit.item.A <- coef(fit.A, IRTpars=T,as.data.frame=T)
  theta.A <- as.vector(fscores(fit.A))
  #hist(theta.A)


  ########################## Unsupervised learning part ##########################
  # Data preparation

  tau.A <- as.data.frame(SecMin.obj[[3]])$Mean
  response.A <- Resp #item response
  Z_sum.A <- SecMin.obj[[7]]  #ltz
  KLD.A <- SecMin.obj[[6]] #KLD index value


  Ht.psf.A <- SecMin.obj[[8]]
  names(Ht.psf.A$PFscores)[names(Ht.psf.A$PFscores) == "PFscores"] <- 'Ht'
  HT.A <- unlist(Ht.psf.A$PFscores)
  HT.A[which(is.na(HT.A))] <- 0.3   ##### suggested by Sijtsma & Molenaar(2002) reported that 0.3 is a critical value for coefficient H


  Lzstar.psf.A <- SecMin.obj[[9]]
  names(Lzstar.psf.A$PFscores)[names(Lzstar.psf.A$PFscores) == "PFscores"] <- 'Lzstart'
  lz.A <- Lzstar.psf.A$PFscores

  NCI.psf.A <- SecMin.obj[[10]]
  names(NCI.psf.A$PFscores)[names(NCI.psf.A$PFscores) == "PFscores"] <- 'NCI'
  NCI.A <- NCI.psf.A$PFscores

  Z_sum.A[is.na(Z_sum.A) == TRUE] <- 600
  KLD.A[which(is.na(KLD.A) == TRUE)] <- 2811
  RT[is.na(RT) == TRUE] <- 0

  colnames(RT) <- paste0("idur.",1:ncol(RT))
  data1.A <- cbind(Flgd,Z_sum.A,HT.A,lz.A,theta.A,tau.A,response.A,RT)
  #names(data1.A) <- make.names(names(data1.A))


  ##################### Feature selection #####################

  correlationMatrix.A <- cor(data1.A)

  # find attributes that are highly corrected (ideally >0.75)
  highlyCorrelated.A <- caret::findCorrelation(correlationMatrix.A, cutoff=0.8)

  if (length(highlyCorrelated.A) > 0) {
    name.highlyCorrelated <- colnames(data1.A)[as.numeric(highlyCorrelated.A)]
    highlyCorrelated <- highlyCorrelated.A - 1
    names(highlyCorrelated) <- paste("Highly Correlated Variable:", name.highlyCorrelated)

    print(highlyCorrelated)

    data <- list(
    selected.features <-  data1.A[,-c(1,as.numeric(highlyCorrelated.A))],
    full.features <-  data1.A[,-1]
    )
    names(data) <- c("selected.features", "full.features")

    return(data)

  } else {
    print("Highly Correlated Variable: None")

    data <- list(
    selected.features <- data1.A[,-1],
    full.features <- data1.A[,-1]
    )
    names(data) <- c("selected.features", "full.features")

    return(data)

    }

}

library(mirt)
data <- prepftr(SecMin.obj, RT, Resp, Flagged, Info = NULL)
