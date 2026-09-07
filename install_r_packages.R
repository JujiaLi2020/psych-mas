# One-time setup: install R packages required for Psych-MAS IRT (mirt, WrightMap, psych).
# Run from project root: Rscript install_r_packages.R
# Or in R: source("install_r_packages.R")

required_versions <- c(
  mirt = "1.46.1",
  aberrance = "0.3.0",
  WrightMap = "1.4",
  psych = "2.6.5"
)
repos <- "https://cloud.r-project.org"

if (!requireNamespace("remotes", quietly = TRUE)) {
  install.packages("remotes", repos = repos)
}

for (p in names(required_versions)) {
  target_version <- required_versions[[p]]
  installed_version <- if (requireNamespace(p, quietly = TRUE)) {
    as.character(packageVersion(p))
  } else {
    NA_character_
  }
  if (is.na(installed_version) || !identical(installed_version, target_version)) {
    remotes::install_version(p, version = target_version, repos = repos, upgrade = "never")
  }
}

# Fail the image build if any required package is still missing.
missing <- names(required_versions)[!vapply(names(required_versions), function(p) requireNamespace(p, quietly = TRUE), logical(1))]
if (length(missing) > 0) {
  message("ERROR: Missing required R packages after install: ", paste(missing, collapse = ", "))
  quit(status = 1)
}

wrong_version <- names(required_versions)[
  vapply(
    names(required_versions),
    function(p) !identical(as.character(packageVersion(p)), required_versions[[p]]),
    logical(1)
  )
]
if (length(wrong_version) > 0) {
  details <- paste(
    wrong_version,
    vapply(wrong_version, function(p) as.character(packageVersion(p)), character(1)),
    "!=",
    required_versions[wrong_version],
    collapse = "; "
  )
  message("ERROR: R package version mismatch: ", details)
  quit(status = 1)
}
