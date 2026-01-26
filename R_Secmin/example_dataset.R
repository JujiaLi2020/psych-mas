#' Example dataset
#'
#' The example dataset contains responses and response times of 1636 test-takers who appeared in a computer-based test 
#' facilitated by a credential program that tests continuously. The dataset contains a total of 69 variables representing 
#' different covariates, responses in each test items, response correctness in each test item, response time for each test item, 
#' and total time taken to complete the test. The test used multiple choice questions with four choice options.
#' The dataset has been carefully selected because it is known to include examinees who engaged in fraudulent test behavior.
#' However, the example dataset is only designed for practices, and may not be interpreted as real data. 
#'
#' @format A data frame with 1636 rows and 69 variables:
#' \describe{
#'   \item{EID}{ID of participants}
#'   \item{Flagged}{Whether the participants were flagged as a cheater. 0=not flagged, 1=flagged}
#'   \item{Pretest}{Id of pretest item set assigned to the candidate}
#'   \item{Attempt}{Count of the attempt number for the candiate. A score of 1 indicates cadidate is a first time examinee}
#'   \item{Country}{Country where candidate was educated}
#'   \item{StateCode}{2-digit code corresponding to the state in which the Candidate applied for licensure}
#'   \item{School_ID}{4-digit code corresponding to the particular institution in which the Candidate received his/her educational training}
#'   \item{cent_id}{4-digit code corresponding to the particular testing center in which the Candidate sat for the exam}
#'   \item{tot_time}{Amount of time taken to complete the test (in seconds)}
#'   \item{iresp.1}{Item responses (1 to 4 or NA) for scored items }
#'   \item{iresp.2}{Item responses (1 to 4 or NA) for scored items }
#'   \item{iresp.3}{Item responses (1 to 4 or NA) for scored items }
#'   \item{iresp.4}{Item responses (1 to 4 or NA) for scored items }
#'   \item{iresp.5}{Item responses (1 to 4 or NA) for scored items }
#'   \item{iresp.6}{Item responses (1 to 4 or NA) for scored items }
#'   \item{iresp.7}{Item responses (1 to 4 or NA) for scored items }
#'   \item{iresp.8}{Item responses (1 to 4 or NA) for scored items }
#'   \item{iresp.9}{Item responses (1 to 4 or NA) for scored items }
#'   \item{iresp.10}{Item responses (1 to 4 or NA) for scored items}
#'   \item{iresp.11}{Item responses (1 to 4 or NA) for scored items}
#'   \item{iresp.12}{Item responses (1 to 4 or NA) for scored items}
#'   \item{iresp.13}{Item responses (1 to 4 or NA) for scored items}
#'   \item{iresp.14}{Item responses (1 to 4 or NA) for scored items}
#'   \item{iresp.15}{Item responses (1 to 4 or NA) for scored items}
#'   \item{iresp.16}{Item responses (1 to 4 or NA) for scored items}
#'   \item{iresp.17}{Item responses (1 to 4 or NA) for scored items}
#'   \item{iresp.18}{Item responses (1 to 4 or NA) for scored items}
#'   \item{iresp.19}{Item responses (1 to 4 or NA) for scored items}
#'   \item{iresp.20}{Item responses (1 to 4 or NA) for scored items}
#'   \item{iraw.1}{Item correct score for scored items. 1= correct, 0= wrong}
#'   \item{iraw.2}{Item correct score for scored items. 1= correct, 0= wrong}
#'   \item{iraw.3}{Item correct score for scored items. 1= correct, 0= wrong}
#'   \item{iraw.4}{Item correct score for scored items. 1= correct, 0= wrong}
#'   \item{iraw.5}{Item correct score for scored items. 1= correct, 0= wrong}
#'   \item{iraw.6}{Item correct score for scored items. 1= correct, 0= wrong}
#'   \item{iraw.7}{Item correct score for scored items. 1= correct, 0= wrong}
#'   \item{iraw.8}{Item correct score for scored items. 1= correct, 0= wrong}
#'   \item{iraw.9}{Item correct score for scored items. 1= correct, 0= wrong}
#'   \item{iraw.10}{Item correct score for scored items. 1= correct, 0= wrong}
#'   \item{iraw.11}{Item correct score for scored items. 1= correct, 0= wrong}
#'   \item{iraw.12}{Item correct score for scored items. 1= correct, 0= wrong}
#'   \item{iraw.13}{Item correct score for scored items. 1= correct, 0= wrong}
#'   \item{iraw.14}{Item correct score for scored items. 1= correct, 0= wrong}
#'   \item{iraw.15}{Item correct score for scored items. 1= correct, 0= wrong}
#'   \item{iraw.16}{Item correct score for scored items. 1= correct, 0= wrong}
#'   \item{iraw.17}{Item correct score for scored items. 1= correct, 0= wrong}
#'   \item{iraw.18}{Item correct score for scored items. 1= correct, 0= wrong}
#'   \item{iraw.19}{Item correct score for scored items. 1= correct, 0= wrong}
#'   \item{iraw.20}{Item correct score for scored items. 1= correct, 0= wrong}
#'   \item{idur.1}{Response time (in seconds) for scored items}
#'   \item{idur.2}{Response time (in seconds) for scored items}
#'   \item{idur.3}{Response time (in seconds) for scored items}
#'   \item{idur.4}{Response time (in seconds) for scored items}
#'   \item{idur.5}{Response time (in seconds) for scored items}
#'   \item{idur.6}{Response time (in seconds) for scored items}
#'   \item{idur.7}{Response time (in seconds) for scored items}
#'   \item{idur.8}{Response time (in seconds) for scored items}
#'   \item{idur.9}{Response time (in seconds) for scored items}
#'   \item{idur.10}{Response time (in seconds) for scored items}
#'   \item{idur.11}{Response time (in seconds) for scored items}
#'   \item{idur.12}{Response time (in seconds) for scored items}
#'   \item{idur.13}{Response time (in seconds) for scored items}
#'   \item{idur.14}{Response time (in seconds) for scored items}
#'   \item{idur.15}{Response time (in seconds) for scored items}
#'   \item{idur.16}{Response time (in seconds) for scored items}
#'   \item{idur.17}{Response time (in seconds) for scored items}
#'   \item{idur.18}{Response time (in seconds) for scored items}
#'   \item{idur.19}{Response time (in seconds) for scored items}
#'   \item{idur.20}{Response time (in seconds) for scored items}
#' }
"example_dataset"
