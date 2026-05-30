# ============================================================
# PsyMAS Tutorial Dataset Simulation
# Semi-simulated computer-based assessment data
# ============================================================

set.seed(2026)

# -----------------------------
# 1. Basic settings
# -----------------------------

N <- 500      # examinees
J <- 40       # items

examinee_id <- paste0("E", sprintf("%03d", 1:N))
item_id <- paste0("I", sprintf("%02d", 1:J))

# -----------------------------
# 2. Generate ability and item parameters
# -----------------------------

theta <- rnorm(N, mean = 0, sd = 1)

a <- rlnorm(J, meanlog = 0, sdlog = 0.20)   # discrimination
b <- rnorm(J, mean = 0, sd = 1)             # difficulty

# 2PL probability
prob <- matrix(NA, nrow = N, ncol = J)

for (i in 1:N) {
  for (j in 1:J) {
    prob[i, j] <- 1 / (1 + exp(-a[j] * (theta[i] - b[j])))
  }
}

responses <- matrix(
  rbinom(N * J, size = 1, prob = as.vector(prob)),
  nrow = N,
  ncol = J
)

rownames(responses) <- examinee_id
colnames(responses) <- item_id

# -----------------------------
# 3. Generate response times
# -----------------------------

tau <- rnorm(N, mean = 0, sd = 0.40)        # person speed
beta <- rnorm(J, mean = 3.4, sd = 0.25)     # item time intensity

log_rt <- matrix(NA, nrow = N, ncol = J)

for (i in 1:N) {
  for (j in 1:J) {
    log_rt[i, j] <- beta[j] - tau[i] + rnorm(1, 0, 0.25)
  }
}

rt <- exp(log_rt)

rownames(rt) <- examinee_id
colnames(rt) <- item_id

# -----------------------------
# 4. Create exposure labels
# -----------------------------

exposed_items <- sample(1:J, size = 8)

item_metadata <- data.frame(
  item_id = item_id,
  position = 1:J,
  a = a,
  b = b,
  beta = beta,
  exposure_status = ifelse(1:J %in% exposed_items, "exposed", "secure")
)

# -----------------------------
# 5. Inject forensic patterns
# -----------------------------

true_group <- rep("normal", N)

# A. Rapid guessing group
rg_ids <- sample(1:N, size = round(N * 0.05))
true_group[rg_ids] <- "rapid_guessing"

for (i in rg_ids) {
  rg_items <- sample(1:J, size = round(J * 0.25))
  responses[i, rg_items] <- rbinom(length(rg_items), size = 1, prob = 0.25)
  rt[i, rg_items] <- runif(length(rg_items), min = 1, max = 6)
}

# B. Preknowledge group
available_ids <- setdiff(1:N, rg_ids)
pk_ids <- sample(available_ids, size = round(N * 0.04))
true_group[pk_ids] <- "preknowledge"

for (i in pk_ids) {
  responses[i, exposed_items] <- rbinom(length(exposed_items), size = 1, prob = 0.90)
  rt[i, exposed_items] <- rt[i, exposed_items] * 0.50
}

# C. Answer similarity / copying pairs
available_ids <- setdiff(1:N, c(rg_ids, pk_ids))
copy_sources <- sample(available_ids, size = 10)
copy_copiers <- sample(setdiff(available_ids, copy_sources), size = 10)

copying_pairs <- data.frame(
  source = examinee_id[copy_sources],
  copier = examinee_id[copy_copiers]
)

for (k in 1:length(copy_sources)) {
  source <- copy_sources[k]
  copier <- copy_copiers[k]
  copied_items <- sample(1:J, size = round(J * 0.45))
  
  responses[copier, copied_items] <- responses[source, copied_items]
  rt[copier, copied_items] <- rt[source, copied_items] * runif(length(copied_items), 0.85, 1.15)
  
  true_group[copier] <- "copying"
}

# D. Mixed evidence group: high ability and fast RT
available_ids <- setdiff(1:N, c(rg_ids, pk_ids, copy_copiers))
mixed_ids <- sample(available_ids, size = round(N * 0.04))
true_group[mixed_ids] <- "mixed_fast_high_ability"

theta[mixed_ids] <- theta[mixed_ids] + 1.25
rt[mixed_ids, ] <- rt[mixed_ids, ] * 0.65

# -----------------------------
# 6. Simulate answer changes
# -----------------------------

initial_responses <- responses
final_responses <- responses

# Normal random answer changes
for (i in 1:N) {
  change_items <- sample(1:J, size = sample(0:3, 1))
  initial_responses[i, change_items] <- 1 - final_responses[i, change_items]
}

# Inject suspicious answer changes for small subset
tamper_ids <- sample(setdiff(1:N, c(rg_ids, pk_ids, copy_copiers, mixed_ids)), size = 10)
true_group[tamper_ids] <- "answer_change"

for (i in tamper_ids) {
  change_items <- sample(1:J, size = round(J * 0.20))
  initial_responses[i, change_items] <- 0
  final_responses[i, change_items] <- 1
}

answer_changes <- data.frame()

for (i in 1:N) {
  for (j in 1:J) {
    answer_changes <- rbind(
      answer_changes,
      data.frame(
        examinee_id = examinee_id[i],
        item_id = item_id[j],
        initial_response = initial_responses[i, j],
        final_response = final_responses[i, j],
        changed = as.integer(initial_responses[i, j] != final_responses[i, j])
      )
    )
  }
}

# Use final responses as the main response matrix
responses <- final_responses

# -----------------------------
# 7. Examinee metadata
# -----------------------------

examinee_metadata <- data.frame(
  examinee_id = examinee_id,
  theta = theta,
  tau = tau,
  true_group = true_group
)

# Optional testing context
testing_context <- data.frame(
  examinee_id = examinee_id,
  test_center = sample(paste0("Center_", 1:10), N, replace = TRUE),
  testing_window = sample(c("Morning", "Afternoon"), N, replace = TRUE),
  seat_group = sample(paste0("Room_", 1:20), N, replace = TRUE)
)

# -----------------------------
# 8. Convert to long format
# -----------------------------

response_long <- data.frame()

for (i in 1:N) {
  temp <- data.frame(
    examinee_id = examinee_id[i],
    item_id = item_id,
    item_position = 1:J,
    response = responses[i, ],
    response_time = rt[i, ],
    log_response_time = log(rt[i, ])
  )
  response_long <- rbind(response_long, temp)
}

response_long <- merge(response_long, item_metadata, by = "item_id")
response_long <- merge(response_long, examinee_metadata, by = "examinee_id")

# -----------------------------
# 9. Save tutorial files
# -----------------------------

dir.create("psymas_tutorial_data", showWarnings = FALSE)

write.csv(responses, "psymas_tutorial_data/responses_matrix.csv", row.names = TRUE)
write.csv(rt, "psymas_tutorial_data/response_times_matrix.csv", row.names = TRUE)
write.csv(item_metadata, "psymas_tutorial_data/item_metadata.csv", row.names = FALSE)
write.csv(examinee_metadata, "psymas_tutorial_data/examinee_metadata.csv", row.names = FALSE)
write.csv(testing_context, "psymas_tutorial_data/testing_context.csv", row.names = FALSE)
write.csv(answer_changes, "psymas_tutorial_data/answer_changes.csv", row.names = FALSE)
write.csv(copying_pairs, "psymas_tutorial_data/copying_pairs.csv", row.names = FALSE)
write.csv(response_long, "psymas_tutorial_data/response_long.csv", row.names = FALSE)

# -----------------------------
# 10. Quick checks
# -----------------------------

table(examinee_metadata$true_group)

head(response_long)
head(item_metadata)
head(answer_changes)
