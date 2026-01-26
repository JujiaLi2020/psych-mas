#' Info dataset
#'
#' The info dataset contains demographic and other covariates for 100 test takers.
#' These covariates may have indirect affects on item response times and item correctness.
#'
#' @format A matrix with 100 rows and 6 variables:
#' \describe{
#'   \item{Attempt}{Count of the attempt number for the candiate. A score of 1 indicates cadidate is a first time examinee}
#'   \item{Country}{Country where candidate was educated}
#'   \item{StateCode}{2-digit code corresponding to the state in which the Candidate applied for licensure}
#'   \item{School_ID}{4-digit code corresponding to the particular institution in which the Candidate received his/her educational training}
#'   \item{cent_id}{4-digit code corresponding to the particular testing center in which the Candidate sat for the exam}
#'   \item{tot_time}{Amount of time taken to complete the test (in seconds)}
#' }
"Info"
