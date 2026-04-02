
#### JAGS setup (required for prim())
## If you see "Unable to call JAGS", either:
## 1. Install JAGS: https://sourceforge.net/projects/mcmc-jags/files/JAGS/ (e.g. JAGS-4.3.0.exe for Windows)
## 2. Then tell runjags where it is. Run once in R before calling prim():
##    runjags.options(jagspath = "C:/Program Files/JAGS/JAGS-4.3.0/x64/bin")   # Windows 64-bit
##    runjags.options(jagspath = "C:/Program Files/JAGS/JAGS-4.3.0/i386/bin")  # Windows 32-bit
##    (Use the path where jags-terminal.exe lives; check with list.files("C:/Program Files/JAGS/"))
## 3. Check: runjags::testjags()

#### data preparation
prepini <- function(RT, Resp) {
  RT.new <- as.matrix(RT)
  RT.new[RT.new== 0] <- NA
  num.RT <- RT.new
  num.RT[is.na(num.RT)] <-1
  logRTs <- log(num.RT)
  
  ini<- list(RT <- RT.new,
             Resp <- as.data.frame(Resp),
             al_hat <- 1/apply(logRTs,2,sd),
             be_hat <- apply(logRTs,2,mean),
             sp_hat <- mean(be_hat) - apply(logRTs,1,mean),
             prec.sp <- 1,
             mu.ipar <- c(mean(al_hat),mean(be_hat)),
             prec.ipar <- diag(1,2,2)
  )
  names(ini) <- c("RT", "Resp", "ini.alpha", "ini.beta", "ini.speed",
                  "ini.prec.speed", "ini.mu.ipar", "ini.prec.ipar")
  
  ini <- list(ini)
  return(ini)

}

### calculating IRT and RT based person-fit indexes
prim <- function(ini) {
  if (!requireNamespace("runjags", quietly = TRUE)) stop("Please install.packages(\"runjags\") and ensure JAGS is installed.")
  if (!requireNamespace("PerFit", quietly = TRUE)) stop("Please install.packages(\"PerFit\")")
  library(runjags)
  library(PerFit)

  # Try to find JAGS on Windows if path not already set
  jags_path <- tryCatch(runjags::runjags.getOption("jagspath"), error = function(e) NA_character_)
  if (identical(.Platform$OS.type, "windows") && (is.na(jags_path) || !nzchar(jags_path))) {
    prog <- Sys.getenv("ProgramFiles", "C:/Program Files")
    prog64 <- Sys.getenv("ProgramW6432", prog)
    for (root in c(prog64, prog)) {
      jags_dir <- file.path(root, "JAGS")
      if (!dir.exists(jags_dir)) next
      for (ver in list.files(jags_dir, full.names = TRUE)) {
        for (sub in c("x64/bin", "bin", "i386/bin")) {
          p <- file.path(ver, sub)
          if (dir.exists(p) && any(grepl("jags", list.files(p), ignore.case = TRUE))) {
            runjags::runjags.options(jagspath = normalizePath(p, mustWork = FALSE))
            break
          }
        }
      }
    }
  }

  # ini: named list with RT, Resp, ini.alpha, ini.beta, ini.speed, ini.prec.speed, ini.mu.ipar, ini.prec.ipar
  RT_dat <- as.matrix(ini[["RT"]])
  N <- nrow(RT_dat)
  m <- ncol(RT_dat)

  model <- "
model {
	for (i in 1:N) {
		for (j in 1:m) {
			RT[i,j] ~ dlnorm(be[j] - sp[i], al[j]^2)
		}
		sp[i] ~ dnorm(0, prec.sp)
	}
	prec.sp ~ dgamma(.001,.001)
	sd.sp <- sqrt(1/prec.sp)

	for (j in 1:m) {
		al[j] <- ipar[j,1]
		be[j] <- ipar[j,2]
		ipar[j,1:2] ~ dmnorm(mu.ipar[1:2], prec.ipar[1:2,1:2])
	}

	mu.ipar[1:2] ~ dmnorm(mu[1:2], prec.ipar[1:2,1:2])
	prec.ipar[1:2,1:2] ~ dwish(V[1:2,1:2], 2)
	var.ipar[1:2,1:2] <- inverse(prec.ipar[1:2,1:2])
}
"

  jags_data <- list(
    RT = RT_dat,
    N = N,
    m = m,
    mu = c(1, 3.5),
    V = matrix(c(5, 0, 0, 5), 2, 2)
  )

  jags_inits <- list(
    sp = ini[["ini.speed"]],
    prec.sp = ini[["ini.prec.speed"]],
    ipar = cbind(ini[["ini.alpha"]], ini[["ini.beta"]]),
    mu.ipar = ini[["ini.mu.ipar"]],
    prec.ipar = ini[["ini.prec.ipar"]]
  )

  out <- run.jags(
    model,
    data = jags_data,
    inits = list(jags_inits, jags_inits),
    n.chains = 2,
    burnin = 1000,
    sample = 8000,
    adapt = 2000,
    monitor = c("al", "be", "mu.ipar", "var.ipar", "sp", "sd.sp")
  )

RT <- as.matrix(ini[["RT"]])
Resp <- as.data.frame(ini[["Resp"]])

out.summary <- extract.runjags(add.summary(out), what = "summary")
n.item <- ncol(RT)
n.person <- nrow(RT)
alpha.table <- out.summary[c(1:n.item),c(1,3:5,9,11)]
beta.table <- out.summary[c((n.item+1):(2*n.item)),c(1,3:5,9,11)]
speed.table <- out.summary[c((nrow(out.summary)-n.person):(nrow(out.summary)-1)),c(1,3:5,9,11)]


##################### RT based PFS calculation ##########################

###### HT index
Ht.psf.A <- PerFit::Ht(Resp)
par(mfrow = c(1,1))
plot(Ht.psf.A)
Ht.cut.A <- PerFit::cutoff(Ht.psf.A,Blvl=.05,Nrep=1000)


###### Lzstar  index
Lzstar.psf.A <- PerFit::lzstar(Resp, IRT.PModel = "1PL")
Lzstar.cut.A <- PerFit::cutoff(Lzstar.psf.A,Blvl=.05,Nrep=1000)
plot(Lzstar.psf.A, cutoff.obj=Lzstar.cut.A,Type="Both",Blv1=0.05,CIlv1=0.9,Xcex=0.8,col.hist="grey",col.area="NA",title="",col.ticks="NA")


###### NCI index
NCI.psf.A <- PerFit::NCI(Resp)
NCI.cut.A <- PerFit::cutoff(NCI.psf.A ,Blvl=.05,Nrep=1000)
plot(NCI.psf.A, Type="Both",Blv1=0.05,CIlv1=0.9,Xcex=0.8,col.hist="grey",col.area="NA",col.ticks="NA",title="")


#### Lt index Marianti & Fox
rt.alpha.A <- as.vector(alpha.table[,3])
rt.beta.A <- as.vector(beta.table[,3])
rt.tau.A <- as.matrix(speed.table[,3])
Form.A.RT <- as.matrix(RT)
I <- ncol(Form.A.RT)
J <- nrow(Form.A.RT)
Z.A=matrix(0,nrow=J,ncol=I)
for(i in 1:I){
  for(j in 1:J){
    Z.A[j,i]<- ((log(Form.A.RT[j,i])-(rt.beta.A[i]-rt.tau.A[j]))*rt.alpha.A[i])^2
  }
}
Z_sum.A <- apply(Z.A,1,sum)
hist(Z_sum.A,col="grey",main = "Distribution of Person Estimates Based on LZ Index",xlab="Lz",breaks=20)
abline(v = qchisq(0.95, ncol(Resp)))
critical_value2 <- qchisq(0.95, ncol(Resp))

# KL index programming

Cheating=Form.A.RT
Cheating[is.na(Cheating)] <- 0
average_cheating <- (apply(Cheating,2,sum)/J)
average_cheating
w2 <- sum(average_cheating)
w2
w_average_cheat <- average_cheating*(1/w2)
w_average_cheat
w1 <- apply(Cheating, 1, sum)
w1[w1 <= 0] <- 1  # avoid division by zero for persons with no RT

KLdata <- diag(1 / w1) %*% Cheating

summary(KLdata)
start.time <- Sys.time()
KLD <- rep(0,J)
for (i in 1:J){
  KLD[i]= sum(w_average_cheat %*% log(w_average_cheat/KLdata[i,]));
}
end.time <- Sys.time()
time.taken <- end.time - start.time
Critical <-quantile(KLD,probs=c(0.5))+1.5*(quantile(KLD,probs=c(0.75))-quantile(KLD,probs=c(0.5)))
Critical
KLD.plot = c(hist(KLD, main = "Distribution of Person Estimates Based on KLD Index"),abline(v=Critical))

LZ.cheating.cases <- which(Z_sum.A>critical_value2)
KLD.cheating.cases <- which(KLD>Critical)


Estimates <- list(alpha.table,
                  beta.table,
                  speed.table,
                  LZ.cheating.cases,
                  KLD.cheating.cases,
                  KLD.value = KLD,
                  Z.score = Z_sum.A,
                  Ht.psf = Ht.psf.A,
                  Lzstar.psf = Lzstar.psf.A,
                  NCI.psf = NCI.psf.A,
                  general = out)

names(Estimates) <- c("alpha", "beta", "speed",
                      "LZ.cheating.cases",
                      "KLD.cheating.cases",
                      "KLD.value", "LZ.value", "Ht.value",
                      "Lzstar.value", "NCI.value", "general")

class(Estimates) <- "SecMin"

return(Estimates)


}

# Example: run after loading data (Resp, RT) and library(runjags), library(PerFit)
# ini <- prepini(RT, Resp)
# SecMin.obj <- prim(ini[[1]])
# install.packages("runjags")  # uncomment if needed
# library(runjags)


