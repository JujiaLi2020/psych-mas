# One-time setup: install R packages required for Psych-MAS IRT (mirt, WrightMap, psych).
# Run from project root: Rscript install_r_packages.R
# Or in R: source("install_r_packages.R")

pkgs <- c("mirt", "WrightMap", "psych", "aberrance")
repos <- "https://cloud.r-project.org"

for (p in pkgs) {
  if (!requireNamespace(p, quietly = TRUE)) {
    install.packages(p, repos = repos)
  }
}
