# R_Secmin — Test Fraud / Cheating Detection

This is an **R project** for detecting test fraud (cheating) in computer-based assessments. It uses **response** (correct/incorrect), **response time (RT)**, IRT and RT-based person-fit indices, and both **supervised** and **unsupervised** machine learning.

---

## Project structure

| File / folder | Purpose |
|---------------|--------|
| **data/** | R data files (`.rda`): `Resp.rda`, `RT.rda`, `Flagged.rda`, `Info.rda`, `example_dataset.rda` |
| **example_dataset.R**, **Resp.R**, **RT.R**, **Flagged.R**, **Info.R** | R documentation for the datasets (variable names, formats). They do **not** load data; use `load("data/...rda")` to load. |
| **datapre_perfits.R** | Core pipeline: data prep (`prepini`), Bayesian RT model + person-fit (`prim`), returns a `SecMin` object. |
| **prepftr.R** | Builds feature sets from a `SecMin` object for ML: `prepftr(SecMin.obj, RT, Resp, Flagged, Info)`. |
| **dv.R** | Diagnostic plots: `dv(RT, Resp)` — RT histograms per item and correct proportion. |
| **plot.SecMin.R** | MCMC/diagnostic plots for a `SecMin` object: `plot.SecMin(SecMin.obj, ...)`. |
| **sprvs.R** | Supervised ML: `sprvs(data, Flagged, ntree)` — KNN and Random Forest for cheating vs non-cheating. |
| **unsprvs.R** | Unsupervised ML: `unsprvs(data, Flagged)` — K-means and FMM for cheating detection. |
| **R_Secmin.Rproj** | RStudio project file. |

---

## Data

- **Resp**: N×J matrix of item correctness (1 = correct, 0 = wrong). Example: 100 persons × 20 items.
- **RT**: N×J matrix of response times (seconds). Same dimensions as `Resp`.
- **Flagged**: length-N vector; 0 = suspected cheater, 1 = non-cheater (used as “truth” for supervised ML).
- **Info**: optional N×I matrix of covariates (e.g. Attempt, Country, tot_time).

---

## How to run (recommended order)

### 1. Set up R and working directory

- Open **R** or **RStudio**.
- Set working directory to the project folder:
  ```r
  setwd("C:/Users/julia/Box/Personal/Research/2026-1-psych-mas/R_Secmin")
  ```

### 2. Install required packages

Run once:

```r
install.packages(c("runjags", "rjags", "PerFit", "mirt", "caret", "randomForest", "class", "mclust", "ggplot2", "plyr"))
```

You also need **JAGS** (Just Another Gibbs Sampler) installed on your system:

- Windows: <https://sourceforge.net/projects/mcmc-jags/files/JAGS/>
- After installing JAGS, restart R and load: `library(runjags)`.

### 3. Load data

```r
load("data/Resp.rda")   # item correctness
load("data/RT.rda")     # response times
load("data/Flagged.rda") # 0 = cheater, 1 = non-cheater
load("data/Info.rda")   # optional covariates
```

Or load the full example dataset:

```r
load("data/example_dataset.rda")
# Then extract Resp, RT, Flagged, Info from the loaded object as needed.
```

### 4. Run the pipeline

**Step A — Data prep and person-fit (SecMin object)**  
In **datapre_perfits.R** you have:

- `prepini(RT, Resp)` → builds initial values `ini`.
- `prim(ini)` → runs the Bayesian lognormal RT model in JAGS and computes person-fit indices (Ht, Lzstar, NCI, LZ, KLD), returns an object of class `SecMin`.

Run in R (after loading `runjags` and `PerFit`):

```r
source("datapre_perfits.R")  # Defines prepini() and prim()
# If the file runs prepini(RT, Resp) and prim(ini) at the bottom, that will create SecMin.obj.
# Otherwise run manually:
ini <- prepini(RT, Resp)
SecMin.obj <- prim(ini[[1]])  # prim expects the inner list
```

Note: `datapre_perfits.R` contains both function definitions and some loose/example code (e.g. `return(Unsupervised.Summary)` and refs to `det.A`, `kmeans.cm`). If you get errors, run only the function definitions (lines for `prepini` and `prim`) and then call `prepini` and `prim` yourself as above.

**Step B — Feature preparation for ML**

```r
source("prepftr.R")
data <- prepftr(SecMin.obj, RT, Resp, Flagged, Info = NULL)
# data$full.features  = all features (no Flagged column)
# data$selected.features = after removing highly correlated variables
```

**Step C — Supervised or unsupervised analysis**

Supervised (KNN + Random Forest):

```r
source("sprvs.R")
sprvs(data$full.features, Flagged, ntree = 150)
# Or: sprvs(data$selected.features, Flagged, ntree = 150)
```

Unsupervised (K-means + FMM):

```r
source("unsprvs.R")
unsprvs(data$full.features, Flagged)
# Or: unsprvs(data$selected.features, Flagged)
```

### 5. Optional

- **RT diagnostics**: `source("dv.R")` then `dv(RT, Resp)` (or call `dv(RT, Resp)` after sourcing).
- **MCMC plots for SecMin**: `plot.SecMin(SecMin.obj, alpha = 1, plot.type = "trace")` (after sourcing `plot.SecMin.R`).

---

## Workflow summary

```
Load data (Resp, RT, Flagged, Info)
       ↓
prepini(RT, Resp) → ini
       ↓
prim(ini) → SecMin.obj  (JAGS + person-fit: Ht, Lzstar, NCI, LZ, KLD)
       ↓
prepftr(SecMin.obj, RT, Resp, Flagged, Info) → data (full.features, selected.features)
       ↓
sprvs(data$..., Flagged, ntree)     and/or     unsprvs(data$..., Flagged)
(KNN + Random Forest)                          (K-means + FMM)
```

---

## Important notes

1. **JAGS**: The core model in `prim()` uses `runjags` and JAGS. Install JAGS and ensure `library(runjags)` works before running `prim()`.
2. **datapre_perfits.R**: Contains mixed code (functions + example/loose lines). If sourcing the whole file causes errors, define only `prepini` and `prim` (e.g. by copying the function blocks into the console or a clean script) and run the pipeline steps manually.
3. **Flagged coding**: In the docs, 0 = cheating, 1 = non-cheating. The supervised/unsupervised code uses this as the reference for confusion matrices and metrics.
4. **R_Secmin.Rproj**: Open this in RStudio to use the project’s working directory and optional workspace restore (`.RData`).

If you tell me your OS and whether you use R or RStudio, I can adapt the run instructions (e.g. exact paths or a single “run all” script).
