#' sprvs
#'
#' sprvs fit gives cheating casess based on supervised machine learning models
#'
#' @param data A required \eqn{N × K} \code{matrix} or \code{data.frame}
#' containing \eqn{K} features of \eqn{N} individuals to be fitted for supervised
#' machine learning models. \code{full.features} and \code{selected.features} produced
#' by \code{\link{prepftr}} can be directly input here.
#' @param Flgd A dichotomous vector marking whether a particular participant is
#' marked as a cheater or not. Non-cheaters need to be marked by 1 and cheaters
#' need to be marked by 0.
#' @param ntree The number of trees to grow of the random forest algorithm. The default number is 150.
#'
#' @import caret
#' @import class
#' @import plyr
#' @import randomForest
#' @import gmodels
#'
#' @return \code{sprvs} returns the results from K-nearest neighbor (KNN) and random forest
#'  \item{KNN Results}{reports the confusion matrix,and several performance metric
#'  including accuracy, kappa, Mcnemar's Test p-value, sensitivity, specificity, positive
#'  predictive value, negative predictive value, prevalence, detection rate, detection
#'  prevalence, and the balanced accuracy from KNN}
#'  \item{Random Forest Results}{reports the confusion matrix,and several performance metric
#'  including accuracy, kappa, Mcnemar's Test p-value, sensitivity, specificity, positive
#'  predictive value, negative predictive value, prevalence, detection rate, detection
#'  prevalence, and the balanced accuracy by random forest}
#' @export


sprvs <- function(data, Flgd, ntree ){

  ### KNN ####
  names(Flgd) <- "Flgd"
  all.data <- cbind(data,Flgd)

  maxs <- apply(all.data, 2, max)
  mins <- apply(all.data, 2, min)
  scaled.A <- as.data.frame(scale(all.data, center = mins, scale = maxs - mins))

  #### random sampling the training all.dataset
  set.seed(1234)
  ind <- sample(2, nrow(all.data), replace=TRUE, prob=c(0.66, 0.34))
  KNN.A.training <- as.data.frame(scaled.A[ind==1,])
  KNN.A.test <-  as.data.frame(scaled.A[ind==2,])

  KNN.A.trainLabels <- all.data$Flgd[ind==1]
  KNN.A.testLabels  <- all.data$Flgd[ind==2]
  set.seed(2234)
  KNN.A.predict <- class::knn(train=KNN.A.training,test=KNN.A.test,cl=KNN.A.trainLabels,k=2,prob = T)
  plot(KNN.A.predict)

  KNN.pred <- as.numeric(KNN.A.predict)
  KNN.pred <- factor(KNN.pred,levels = c(1,2),labels = c("cheated","not cheated"))
  KNN.testlabels <- factor(KNN.A.testLabels,levels = c(0,1),labels = c("cheated","not cheated"))
  KNN.cm <- caret::confusionMatrix(KNN.pred,KNN.testlabels)

  ### Random forest #####
  set.seed(1234)
  ind.rf <- sample(2, nrow(all.data), replace=TRUE, prob=c(0.66, 0.34))

  RF.A.training <- as.data.frame(scaled.A[ind.rf ==1,])
  RF.A.test <-  as.data.frame(scaled.A[ind.rf ==2,])

  RF.A.training <- cbind(RF.A.training[!sapply(RF.A.training, is.list)],
                         (t(apply(RF.A.training[sapply(RF.A.training, is.list)], 1, unlist))))
  names(RF.A.training) <- make.names(names(RF.A.training))
  RF.A.trainLabels <- all.data$Flgd[ind.rf==1]
  RF.A.testLabels <- all.data$Flgd[ind.rf==2]

  varNames <- names(RF.A.training)
  # Exclude ID or Response variable
  varNames <- varNames[!varNames %in% c("Flgd")]
  # add + sign between exploratory variables
  varNames1 <- paste(varNames, collapse = "+")
  # Add response variable and convert to a formula object
  rf.form <- as.formula(paste("as.factor(Flgd)", varNames1, sep = " ~ "))

  set.seed(2234)
  A.rf <- randomForest::randomForest(rf.form,as.matrix(RF.A.training), ntree = ntree, importance=T,na.action = na.omit, replace=F)
  plot(x=c(1:ntree),y=A.rf$err.rate[,1], xlab="Number of Trees", ylab="Error", type="n", main = "OOB Errors - Random Forest")
  lines(x=c(1:ntree),y=A.rf$err.rate[,1], xlab="Number of Trees", ylab="Error", col=2, lty=1, lwd=2)


  RF.A.test <- cbind(RF.A.test[!sapply(RF.A.test, is.list)],
                     (t(apply(RF.A.test[sapply(RF.A.test, is.list)], 1, unlist))))
  names(RF.A.test) <- make.names(names(RF.A.test))
  pred.A <- predict(A.rf, RF.A.test, probability=F)

  RF.pred <- as.numeric(pred.A)
  RF.pred <- factor(RF.pred,levels = c(1,2),labels = c("cheated","not cheated"))
  RF.testlabels <- factor(RF.A.testLabels,levels = c(0,1),labels = c("cheated","not cheated"))
  RF.cm <- caret::confusionMatrix(RF.pred, RF.testlabels)

  # Determine the variable importance measures

  A.rf.importance<-round(randomForest::importance(A.rf), 2)
  randomForest::varImpPlot(A.rf, type=2, col="darkred", sort=TRUE, n.var=40,cex=0.8,main = "Variable Importance - Random Forest")

  Supervised.Summary <- (list(
    "KNN Results" = KNN.cm,
    "Random Forest Results" = RF.cm))

  return(Supervised.Summary)

}

sprvs(data[[2]],Flagged, ntree = 15)

      