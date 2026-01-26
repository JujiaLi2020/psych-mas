#' unsprvs
#'
#' unsprvs gives cheating cases based on unsupervised machine learning models
#'
#' @param data A required \eqn{N × K} \code{matrix} or \code{data.frame}
#' containing \eqn{K} features of \eqn{N} individuals to be fitted for unsupervised
#' machine learning models. \code{full.features} and \code{selected.features} produced
#' by \code{\link{prepftr}} can be directly input here.
#' @param Flgd A dichotomous vector marking whether a particular participant is
#' marked as a cheater or not. Non-cheaters need to be marked by 1 and cheaters
#' need to be marked by 0.
#'
#' @import caret
#' @import mlbench
#' @import mclust
#' @importFrom dplyr case_when
#' @import stats
#'
#' @return \code{unsprvs} returns the results from Kmeans and finite mixture
#' modeling (FMM)
#'  \item{Detected Cheating Cases Using Kmeans}{reports the cheating cases detected
#'  by Kmeans}
#'  \item{Kmeas Results}{reports the confusion matrix,and several performance metric
#'  including accuracy, kappa, Mcnemar's Test p-value, sensitivity, specificity, positive
#'  predictive value, negative predictive value, prevalence, detection rate, detection
#'  prevalence, and the balanced accuracy from Kmeans analysis}
#'  \item{Detected Cheating Cases Using FMM}{reports
#'  the cheating cases detected by FMM}
#'  \item{FMM Results}{reports the confusion matrix,and several performance metric
#'  including accuracy, kappa, Mcnemar's Test p-value, sensitivity, specificity, positive
#'  predictive value, negative predictive value, prevalence, detection rate, detection
#'  prevalence, and the balanced accuracy by FMM}
#' @export

unsprvs <- function(data,Flgd){

  ID.A <- seq(1:length(data[,1]))
  Flgd <- as.vector(unlist(Flgd))
  Flgd <- factor(Flgd,levels = c(0,1),labels = c("cheated","not cheated"))

  ### KMeans #####
  set.seed(520)
  km.out.A <- kmeans(data,2,nstart=2)
  km.out.A$size #kmeans cluster size

  #### calculate the sensitivity
  prop.cluster <- table(km.out.A$cluster)
  det.A <- as.vector(which(km.out.A$cluster== which(prop.cluster < length(data[,1])/2)))

  #not.Cheater.ID <- ID.A[-c(cheat.key.A)]
  not.det.A <- as.vector(which(km.out.A$cluster== which(prop.cluster >= length(data[,1])/2)))

  #### generate a confusion matrix as a function
  kmeans.pred <- km.out.A$cluster
  kmeans.pred <- factor(kmeans.pred,levels = c(1,2),labels = c("cheated","not cheated"))
  kmeans.cm <- caret::confusionMatrix(kmeans.pred,Flgd)

  k.max <- 5
  wss.A <- sapply(1:k.max,function(k){kmeans(data,k,nstart=20)$tot.withinss})

  ### Mixture mutivaraite normal  ######
  set.seed(520)

  fit.cluster.A <- mclust::Mclust(data,G=1:2)

  BIC <- NULL

  for(j in 1:5)
  {
    rBIC <- mclust::mclustBIC(data, verbose = T,
                      initialization = list(hcPairs = mclust::hcRandomPairs(data)))
    BIC <- mclust::mclustBICupdate(BIC, rBIC)
  }


  prop.cluster.m <- table(fit.cluster.A$classification)
  det.m.A <- which(km.out.A$cluster== which(prop.cluster.m < length(data[,1])/2))


  # confusion matrix


  mixmul.pred <- fit.cluster.A$classification
  mixmul.pred <- dplyr::case_when(
    mixmul.pred == 1 ~ 2,
    mixmul.pred == 2 ~ 1
  )
  mixmul.pred <- factor(mixmul.pred,levels = c(1,2), labels = c("cheated","not cheated"))
  mixmul.cm <- caret::confusionMatrix(mixmul.pred,Flgd)

  Unsupervised.Summary <- (list(
    "Detected Cheating Cases Using Kmeans" = det.A,
    "Kmeas Results" = kmeans.cm,
    "Detected Cheating Cases Using FMM" = det.m.A,
    "FMM Results" = mixmul.cm))

  return(Unsupervised.Summary)

}

library(mclust)
unsprvs(data[[2]],Flagged)


