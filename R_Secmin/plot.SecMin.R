#' plot.SecMin
#'
#' plot.SecMin fit the lognormal model and return parameter estimates
#'
#' @param x objects from class \code{SecMin}
#' @param alpha item numbers of the alpha parameter interested
#' @param beta item numbers of the alpha parameter interested
#' @param speed person IDs of the speed parameter interested
#' @param plot.type plot types
#' @param ... graphical parameters of \code{plot()}
#'
#' @import ggplot2
#' @import runjags
#'
#'
#' @return trace, histrogram, autocorrelation and ECDF plots of alpha, beta, and speed parameters
#' @export


plot.SecMin <-  function(x, alpha = NULL, beta = NULL, speed = NULL,
                       plot.type = c('trace','histogram','autocorr','ecdf'), ...) {
  if (is.null(alpha) == T && is.null(beta) == T && is.null(speed) == T) {
    plot(x[[11]], plot.type = plot.type)
  }else if(is.null(alpha) == F && is.null(beta) == T && is.null(speed) == T) {
    plot(x[[11]],vars = paste0("'al[", alpha, "]'"), plot.type = plot.type)
  }else if(is.null(alpha) == T && is.null(beta) == F && is.null(speed) == T) {
    plot(x[[11]],vars = paste0("'be[", beta, "]'"), plot.type = plot.type)
  }else if(is.null(alpha) == T && is.null(beta) == T && is.null(speed) == F) {
    plot(x[[11]],vars = paste0("'sp[", speed, "]'"), plot.type = plot.type)
  }else {stop("plot.SecMin is only applicable for single type of parameter.", call. = FALSE)
  }

}


