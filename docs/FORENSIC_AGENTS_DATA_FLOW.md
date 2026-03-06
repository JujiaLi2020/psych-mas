# How forensic agents read and process data

All agents receive a shared **State** dict (from the backend or the Preparation workflow) and return a **flags** fragment that gets merged into the result. Data flow is: **State → Python (pandas/numpy) → R (via rpy2)** using only list/vector handoff to avoid numpy→R conversion errors.

---

## Shared state (inputs)

| Key | Type | Source | Used by |
|-----|------|--------|---------|
| `responses` | list of dicts (or list of lists) | UI upload → backend `/detect` payload | All agents that need item scores |
| `rt_data` | list of dicts (or list of lists) | UI upload → backend payload | pm_agent, rg_agent (when RT-based methods run) |
| `psi_data` / `item_params` | list of dicts (a, b, g/c per item) | Preparation (IRT) → backend payload | pm_agent, ac_agent, pk_agent |
| `compromised_items` | list of int (1-based item indices) | UI input (Preparation) → backend payload | pk_agent only |
| `aberrance_functions` | list of str (e.g. `["detect_nm","detect_pk"]`) | UI checkboxes → backend payload | All agents (to skip if not selected) |

---

## Per-agent flow

### 1. **nm_agent** (Nonparametric misfit — detect_nm)

- **Reads:** `state["responses"]`, `state["aberrance_functions"]`
- **Process:**
  1. `resp_df = pd.DataFrame(state["responses"])`, then `keep_cols = _forensic_keep_cols(resp_df)` (dichotomous 0/1 columns only).
  2. `_forensic_init_r(resp_df, keep_cols)` builds the response matrix **x** in R from a **pure Python list** (`x_vec` → `matrix(..., byrow=TRUE)`). No DataFrame or numpy is passed to R.
  3. R: `detect_nm(method = c('G_S', 'NC_S', ...), x = x)`.
  4. Results read back via `ro.r("nm_out$stat")` and column-wise `as.numeric(...)`; converted to Python with `rpy2py`; flags from ZU3_S &lt; -2 and HT_S ≤ 5th percentile.
- **Returns:** `flags["nm_agent"]`: `stat`, `methods`, `flagged`, `n_persons`.

---

### 2. **pm_agent** (Parametric misfit — detect_pm)

- **Reads:** `state["responses"]`, `state["psi_data"]` (or `item_params`), `state["rt_data"]` (optional), `state["aberrance_functions"]`
- **Process:**
  1. Same as nm_agent: `resp_df`, `keep_cols`, then `_forensic_init_r` → **x** in R.
  2. `ip_df = pd.DataFrame(psi_src)`; `_forensic_build_psi(ro, ip_df, len(keep_cols))` pushes **psi** (a, b, c) into R via `FloatVector(a_vals)` etc. and `ro.r("psi <- as.matrix(cbind(...))")`.
  3. If RT present: build **y** (log RT) in R from a **Python list**: `y_flat = np.log(...).flatten().tolist()` → `ro.FloatVector(y_flat)` → `y <- matrix(...)`; then extend psi with alpha/beta in R.
  4. R: `detect_pm(..., psi = psi, x = x, y = y)` or `(..., psi = psi, x = x)`.
  5. Stat/pval/flag read back column-wise from R and converted to Python.
- **Returns:** `flags["pm_agent"]`: `stat`, `methods`, `flagged`, `n_persons`.

---

### 3. **ac_agent** (Answer copying — detect_ac)

- **Reads:** `state["responses"]`, `state["psi_data"]`, `state["aberrance_functions"]`
- **Process:**
  1. Same as above: `_forensic_init_r` → **x**, `_forensic_build_psi` → **psi** in R.
  2. R: `detect_ac(method = c('OMG_S','GBT_S'), psi = psi, x = x, alpha = 0.05)`.
  3. Pairs and stats read back via `as.data.frame(ac_out$stat)` and `rpy2py`; pairs built in Python as (Source, Copier) from R’s combn order.
- **Returns:** `flags["ac_agent"]`: `pairs`, `stat`, `flagged`, `flagged_copiers`, etc.

---

### 4. **as_agent** (Answer similarity — detect_as)

- **Reads:** `state["responses"]`, `state["aberrance_functions"]`
- **Process:**
  1. `_forensic_init_r` → **x** in R only (no psi).
  2. R: `detect_as(method = 'M4_S', x = x, alpha = 0.05)`.
  3. Stat/pval/flag read back; Python builds pair list and filters by p-value / flag.
- **Returns:** `flags["as_agent"]`: `stat`, `methods`, `flagged_pairs`, etc.

---

### 5. **rg_agent** (Rapid guessing — detect_rg)

- **Reads:** `state["responses"]`, `state["rt_data"]`, `state["aberrance_functions"]`
- **Process:**
  1. `_forensic_init_r` → **x** in R.
  2. RT matrix **t** built in R from a **Python list**: `flat = t_block.values.flatten().tolist()` → `ro.FloatVector(flat)` → `t <- matrix(...)`.
  3. R: `detect_rg(method = 'NT', t = t, x = x, nt = 10)`.
  4. RTE and flag read back with `rpy2py`.
- **Returns:** `flags["rg_agent"]`: `rte`, `flagged`, `methods`.

---

### 6. **cp_agent** (Change point — detect_cp)

- **Reads:** `state["responses"]`, `state["aberrance_functions"]` (runs when detect_rg is selected).
- **Process:**
  1. `_forensic_init_r` → **x** in R.
  2. R: `detect_cp(method = 'MCP', x = x)`.
  3. Stat and flag read back.
- **Returns:** `flags["cp_agent"]`: `stat`, `flagged`, `methods`.

---

### 7. **tt_agent** (Test tampering — detect_tt)

- **Reads:** `state["aberrance_functions"]` only.
- **Process:** No R call; erasure data not in current workflow.
- **Returns:** `flags["tt_agent"]`: `info` message, empty `flagged`, `methods`.

---

### 8. **pk_agent** (Preknowledge — detect_pk)

- **Reads:** `state["responses"]`, `state["psi_data"]`, `state["compromised_items"]`, `state["aberrance_functions"]`
- **Process:**
  1. `_forensic_init_r` → **x** in R (pure Python list → `IntVector` → `matrix`).
  2. `_forensic_build_psi` → **psi** in R (FloatVector from Python lists).
  3. **ci** in R: `ro.r("c(" + ",".join(map(str, ci_1based)) + ")")` (no Python object assigned).
  4. R: `detect_pk(method = c('L_S','S_S','W_S'), ci = ci, psi = psi, x = x, alpha = 0.05)`.
  5. Stat and flag read back with `rpy2py`.
- **Returns:** `flags["pk_agent"]`: `stat`, `flagged`, `methods`.

---

## Helpers (shared)

- **`_forensic_keep_cols(resp_df)`**  
  Returns column names that are valid dichotomous (0/1) with both values present.

- **`_forensic_init_r(resp_df, keep_cols)`**  
  Loads `aberrance`, builds **x** in R from a **pure Python list** (loop over `block.iloc[i,j]`, append `int(...)`), then `ro.IntVector(x_flat)` and `x <- matrix(...)`. No DataFrame or numpy is passed to R.

- **`_forensic_build_psi(ro, ip_df, n_items)`**  
  Builds **psi** in R from item params: `a_vals`, `b_vals`, `c_vals` as Python lists → `ro.FloatVector(...)` → `psi <- as.matrix(cbind(...))`.

---

## Rule to avoid "py2rpy not defined for numpy.ndarray"

- **Never** assign a pandas DataFrame or a numpy array to R (e.g. `ro.globalenv["x"] = df` or `= arr`).
- **Always** build R objects from **Python scalars/lists** or from R code:
  - Response matrix: Python list of ints → `ro.IntVector(x_flat)` → `x <- matrix(...)`.
  - RT matrix: Python list of floats → `ro.FloatVector(flat)` → `t <- matrix(...)`.
  - Psi: Python lists → `ro.FloatVector(a_vals)` etc. → `psi <- as.matrix(cbind(...))`.
- Optionally call **`numpy2ri.activate()`** (e.g. in backend and in `_forensic_init_r`) so any remaining numpy is converted when possible.
