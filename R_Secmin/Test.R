library(runjags)

install.packages("runjags")
install.packages("PerFit")

load("data/Resp.rda")
load("data/RT.rda")
load("data/Flagged.rda")
load("data/Info.rda")

ini <- prepini(RT, Resp)
SecMin.obj <- prim(ini[[1]])


source("prepftr.R")   # defines prepftr() if not already sourced

# You need RT, Resp, Flagged in your workspace (from load("data/...rda"))
data <- prepftr(SecMin.obj, RT, Resp, Flagged, Info = NULL)

data$full.features

data$selected.features


source("sprvs.R")
sprvs(data$full.features, Flagged, ntree = 150)
# Or: sprvs(data$selected.features, Flagged, ntree = 150)

source("unsprvs.R")
unsprvs(data$full.features, Flagged)
# Or: unsprvs(data$selected.features, Flagged)


#3. Optional: plots and diagnostics
#RT diagnostics:
source("dv.R") 
dv(RT, Resp)

#MCMC / SecMin plots:
source("plot.SecMin.R")
plot(SecMin.obj, alpha = 1, plot.type = "trace")











# Install once (if needed)
# install.packages("aberrance")

library(aberrance)

set.seed(123)

# --- 1. Simulate toy data -------------------------------------------------

N  <- 100   # persons
J  <- 20    # items

# Dichotomous responses: 0/1
x <- matrix(rbinom(N * J, size = 1, prob = 0.7), nrow = N, ncol = J)

# Response times (in seconds), log-normal-ish
y <- matrix(rlnorm(N * J, meanlog = 1.5, sdlog = 0.5), nrow = N, ncol = J)

# --- 2. Nonparametric person-fit (score-based indices) --------------------

nm_out <- detect_nm(
  method = c(
    "G_S","NC_S","U1_S","U3_S",
    "ZU3_S","A_S","D_S","E_S",
    "C_S","MC_S","PC_S","HT_S"
  ),
  x = x
)

cat("Nonparametric detect_nm methods:\n")
print(colnames(nm_out$stat))
cat("\nFirst 6 rows of stat matrix:\n")
print(head(nm_out$stat))

# Optional: p-values if provided
if (!is.null(nm_out$pval)) {
  cat("\nFirst 6 rows of p-value matrix:\n")
  print(head(nm_out$pval))
}

# --- 3. RT-based nonparametric (KL_T) -------------------------------------

kl_out <- detect_nm(method = "KL_T", y = y)

cat("\nKL_T results (first 6 rows):\n")
print(head(kl_out$stat))

# --- 4. Quick summary of how many students flagged ------------------------

cat("\nNumber of persons flagged (any nonparametric index):\n")
print(sum(apply(nm_out$flag, 1, any)))

cat("\nNumber of persons flagged by KL_T:\n")
print(sum(kl_out$flag))










