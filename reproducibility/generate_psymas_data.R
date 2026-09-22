# ============================================================
# PsyMAS Tutorial Dataset Simulation
# Semi-simulated computer-based assessment data
# ============================================================
#
# Purpose:
#   Generate a semi-simulated dataset for demonstrating PsyMAS,
#   a human-in-the-loop psychometric forensics workbench.
#   test security review.
#
# Design:
#   - 500 examinees
#   - 40 dichotomous items
#   - Final item scores
#   - Initial item scores for answer-change evidence
#   - Response times
#   - Exposed / compromised item labels
#   - Embedded forensic review scenarios
#
# Important design choice:
#   This script creates simulation input data and records the generating
#   parameters. The saved v0.7.7 demonstration supplied item parameters derived
#   from these simulation parameters; it did not re-estimate them with mirt.
#   For new runs, PsyMAS uses mirt only when item_params.csv is absent.
#   The prototype also derives data-availability indicators
#   from uploaded tables.
#
# ============================================================

set.seed(2026)

# -----------------------------
# 1. Basic settings
# -----------------------------

N <- 500
J <- 40

examinee_id <- paste0("E", sprintf("%03d", 1:N))
item_id <- paste0("I", sprintf("%02d", 1:J))

# -----------------------------
# 2. Generate true latent traits and item parameters
# -----------------------------
#
# These parameters are used only to generate the simulated data.
# In the PsyMAS prototype, item and examinee parameters are
# re-estimated from the final score matrix using mirt.

theta_true <- rnorm(N, mean = 0, sd = 1)
speed_true <- rnorm(N, mean = 0, sd = 0.40)

true_a <- rlnorm(J, meanlog = 0, sdlog = 0.20)
true_b <- rnorm(J, mean = 0, sd = 1)
true_beta <- rnorm(J, mean = 3.4, sd = 0.25)

# -----------------------------
# 3. Generate final item scores under a 2PL model
# -----------------------------

prob <- matrix(NA, nrow = N, ncol = J)

for (i in 1:N) {
  for (j in 1:J) {
    prob[i, j] <- 1 / (1 + exp(-true_a[j] * (theta_true[i] - true_b[j])))
  }
}

final_scores <- matrix(
  rbinom(N * J, size = 1, prob = as.vector(prob)),
  nrow = N,
  ncol = J
)

rownames(final_scores) <- examinee_id
colnames(final_scores) <- item_id

# -----------------------------
# 4. Generate response times
# -----------------------------
#
# Raw response times are saved as the required input.
# Log response times can be derived by the prototype.

log_rt <- matrix(NA, nrow = N, ncol = J)

for (i in 1:N) {
  for (j in 1:J) {
    log_rt[i, j] <- true_beta[j] - speed_true[i] + rnorm(1, 0, 0.25)
  }
}

response_times <- exp(log_rt)

rownames(response_times) <- examinee_id
colnames(response_times) <- item_id

# -----------------------------
# 5. Create exposed / compromised item labels
# -----------------------------
#
# Exposed items are selected from relatively difficult items
# so that fast correct responses on these items provide a clearer
# preknowledge-like pattern.

difficult_pool <- which(true_b > median(true_b))

if (length(difficult_pool) >= 8) {
  exposed_items <- sample(difficult_pool, size = 8)
} else {
  exposed_items <- sample(1:J, size = 8)
}

item_metadata <- data.frame(
  item_id = item_id,
  item_position = 1:J,
  true_a = true_a,
  true_b = true_b,
  true_beta = true_beta,
  exposure_status = ifelse(1:J %in% exposed_items, "exposed", "secure")
)

compromised_items <- item_metadata[item_metadata$exposure_status == "exposed", ]
compromised_items$compromised_flag <- 1

# -----------------------------
# 6. Initialize scenario labels
# -----------------------------

true_group <- rep("normal", N)
assigned_ids <- integer(0)

sample_available <- function(exclude_ids, size) {
  available <- setdiff(1:N, exclude_ids)
  sample(available, size = size)
}

# -----------------------------
# 7. Scenario A: Rapid guessing
# -----------------------------
#
# Rapid guessing examinees respond very quickly on selected items
# and have low accuracy on those items.

rg_ids <- sample_available(assigned_ids, size = round(N * 0.05))
assigned_ids <- c(assigned_ids, rg_ids)

true_group[rg_ids] <- "rapid_guessing"

for (i in rg_ids) {
  rg_items <- sample(1:J, size = round(J * 0.25))
  
  final_scores[i, rg_items] <- rbinom(length(rg_items), size = 1, prob = 0.25)
  response_times[i, rg_items] <- runif(length(rg_items), min = 1, max = 6)
}

# -----------------------------
# 8. Scenario B: Possible item preknowledge
# -----------------------------
#
# Preknowledge-like examinees respond quickly and correctly on
# exposed difficult items. This simulates a review scenario, not
# confirmed misconduct.

pk_ids <- sample_available(assigned_ids, size = round(N * 0.04))
assigned_ids <- c(assigned_ids, pk_ids)

true_group[pk_ids] <- "preknowledge"

for (i in pk_ids) {
  final_scores[i, exposed_items] <- rbinom(length(exposed_items), size = 1, prob = 0.90)
  response_times[i, exposed_items] <- response_times[i, exposed_items] * 0.50
}

# -----------------------------
# 9. Scenario C: Answer similarity / copying
# -----------------------------
#
# Copier examinees copy part of the response pattern from source
# examinees. The copying-pair table is a simulation truth table
# used for validation, not an operational input.

copy_sources <- sample_available(assigned_ids, size = 10)
assigned_ids <- c(assigned_ids, copy_sources)

copy_copiers <- sample_available(assigned_ids, size = 10)
assigned_ids <- c(assigned_ids, copy_copiers)

true_group[copy_sources] <- "copying_source"
true_group[copy_copiers] <- "copying_copier"

copying_pairs_truth <- data.frame(
  source_id = examinee_id[copy_sources],
  copier_id = examinee_id[copy_copiers],
  scenario_label = "score_based_copying"
)

for (k in seq_along(copy_sources)) {
  source <- copy_sources[k]
  copier <- copy_copiers[k]
  
  copied_items <- sample(1:J, size = round(J * 0.45))
  
  final_scores[copier, copied_items] <- final_scores[source, copied_items]
  
  # Timing similarity is added to create a stronger demonstration case.
  response_times[copier, copied_items] <- 
    response_times[source, copied_items] * runif(length(copied_items), 0.85, 1.15)
}

# -----------------------------
# 10. Scenario D: Mixed fast high-ability performance
# -----------------------------
#
# These examinees have higher true ability and faster response times.
# This scenario may produce ambiguous evidence because fast responses
# may be explained by higher ability rather than security risk.

mixed_ids <- sample_available(assigned_ids, size = round(N * 0.04))
assigned_ids <- c(assigned_ids, mixed_ids)

true_group[mixed_ids] <- "mixed_fast_high_ability"

for (i in mixed_ids) {
  theta_true[i] <- theta_true[i] + 1.25
  
  for (j in 1:J) {
    p_new <- 1 / (1 + exp(-true_a[j] * (theta_true[i] - true_b[j])))
    final_scores[i, j] <- rbinom(1, size = 1, prob = p_new)
  }
  
  response_times[i, ] <- response_times[i, ] * 0.65
}

# -----------------------------
# 11. Generate initial scores and answer changes
# -----------------------------
#
# Initial scores represent the response state before answer changes.
# Final scores represent the response state after answer changes.

initial_scores <- final_scores

# Normal random answer changes
for (i in 1:N) {
  change_items <- sample(1:J, size = sample(0:3, 1))
  initial_scores[i, change_items] <- 1 - final_scores[i, change_items]
}

# -----------------------------
# 12. Scenario E: Suspicious answer changes
# -----------------------------
#
# These examinees show many wrong-to-right changes.
# This supports a prototype answer-change module.
# It does not fully simulate distractor-based test tampering.

answer_change_ids <- sample_available(assigned_ids, size = 10)
assigned_ids <- c(assigned_ids, answer_change_ids)

true_group[answer_change_ids] <- "answer_change"

for (i in answer_change_ids) {
  change_items <- sample(1:J, size = round(J * 0.20))
  
  initial_scores[i, change_items] <- 0
  final_scores[i, change_items] <- 1
}

rownames(initial_scores) <- examinee_id
colnames(initial_scores) <- item_id
rownames(final_scores) <- examinee_id
colnames(final_scores) <- item_id

# -----------------------------
# 13. Scenario F: Limited evidence due to missing response times
# -----------------------------
#
# Missing RT cases allow the prototype to demonstrate unavailable
# evidence handling. The prototype should identify missing RT evidence
# automatically during data preparation.

missing_rt_ids <- sample_available(assigned_ids, size = 15)
assigned_ids <- c(assigned_ids, missing_rt_ids)

true_group[missing_rt_ids] <- "missing_rt_limited_evidence"

for (i in missing_rt_ids) {
  missing_items <- sample(1:J, size = round(J * 0.30))
  response_times[i, missing_items] <- NA
}

# -----------------------------
# 14. Build answer-change long table
# -----------------------------

answer_changes_long <- data.frame()

for (i in 1:N) {
  for (j in 1:J) {
    answer_changes_long <- rbind(
      answer_changes_long,
      data.frame(
        examinee_id = examinee_id[i],
        item_id = item_id[j],
        item_position = j,
        initial_score = initial_scores[i, j],
        final_score = final_scores[i, j],
        changed = as.integer(initial_scores[i, j] != final_scores[i, j]),
        wrong_to_right = as.integer(initial_scores[i, j] == 0 & final_scores[i, j] == 1),
        right_to_wrong = as.integer(initial_scores[i, j] == 1 & final_scores[i, j] == 0)
      )
    )
  }
}

# -----------------------------
# 15. Build examinee metadata
# -----------------------------

examinee_metadata <- data.frame(
  examinee_id = examinee_id,
  true_theta = theta_true,
  true_speed = speed_true,
  true_group = true_group
)

scenario_key <- data.frame(
  examinee_id = examinee_id,
  true_group = true_group,
  scenario_type = ifelse(true_group == "normal", "baseline", "embedded_review_scenario")
)

scenario_description <- data.frame(
  true_group = c(
    "normal",
    "rapid_guessing",
    "preknowledge",
    "copying_source",
    "copying_copier",
    "mixed_fast_high_ability",
    "answer_change",
    "missing_rt_limited_evidence"
  ),
  description = c(
    "Normal response pattern",
    "Low accuracy and very short response time on selected items",
    "High accuracy and fast response time on exposed difficult items",
    "Source examinee in a simulated copying pair",
    "Copier examinee with unusually similar score pattern",
    "High-ability examinee with fast response time; may create mixed evidence",
    "Many wrong-to-right answer changes",
    "Partial missing response-time data for unavailable-evidence handling"
  )
)

scenario_summary <- as.data.frame(table(true_group))
colnames(scenario_summary) <- c("true_group", "n_examinees")

scenario_summary <- merge(
  scenario_summary,
  scenario_description,
  by = "true_group",
  all.x = TRUE
)

# -----------------------------
# 16. Optional testing context
# -----------------------------
#
# This table provides simple contextual metadata.
# It is not meant to simulate real proctoring notes, video, device logs,
# or other multimodal process data.

testing_context <- data.frame(
  examinee_id = examinee_id,
  test_center = sample(paste0("Center_", 1:10), N, replace = TRUE),
  testing_window = sample(c("Morning", "Afternoon"), N, replace = TRUE),
  room_id = sample(paste0("Room_", 1:20), N, replace = TRUE),
  seat_group = sample(paste0("SeatGroup_", 1:30), N, replace = TRUE)
)

# -----------------------------
# 17. Optional long-format response table
# -----------------------------
#
# This file is useful for inspection, dashboard display, and LLM-assisted
# table recognition. Most aberrance functions use matrix inputs instead.

response_long <- data.frame()

for (i in 1:N) {
  temp <- data.frame(
    examinee_id = examinee_id[i],
    item_id = item_id,
    item_position = 1:J,
    final_score = final_scores[i, ],
    response_time = response_times[i, ]
  )
  response_long <- rbind(response_long, temp)
}

response_long <- merge(response_long, item_metadata, by = c("item_id", "item_position"))
response_long <- merge(response_long, examinee_metadata, by = "examinee_id")
response_long <- merge(response_long, testing_context, by = "examinee_id")

# -----------------------------
# 18. Quick descriptive checks
# -----------------------------

group_check <- aggregate(
  cbind(final_score, response_time) ~ true_group,
  data = response_long,
  FUN = function(x) mean(x, na.rm = TRUE)
)

colnames(group_check) <- c("true_group", "mean_score", "mean_response_time")

answer_change_summary <- aggregate(
  cbind(changed, wrong_to_right, right_to_wrong) ~ examinee_id,
  data = answer_changes_long,
  FUN = sum
)

answer_change_summary <- merge(
  answer_change_summary,
  examinee_metadata[, c("examinee_id", "true_group")],
  by = "examinee_id"
)

# -----------------------------
# 19. Save simulation input files
# -----------------------------

output_dir <- "psymas_tutorial_data"
dir.create(output_dir, showWarnings = FALSE)

# Core input files
write.csv(final_scores, file.path(output_dir, "final_scores_matrix.csv"), row.names = TRUE)
write.csv(initial_scores, file.path(output_dir, "initial_scores_matrix.csv"), row.names = TRUE)
write.csv(response_times, file.path(output_dir, "response_times_matrix.csv"), row.names = TRUE)

# Metadata and scenario files
write.csv(item_metadata, file.path(output_dir, "item_metadata.csv"), row.names = FALSE)
write.csv(examinee_metadata, file.path(output_dir, "examinee_metadata.csv"), row.names = FALSE)
write.csv(compromised_items, file.path(output_dir, "compromised_items.csv"), row.names = FALSE)
write.csv(testing_context, file.path(output_dir, "testing_context.csv"), row.names = FALSE)
write.csv(scenario_key, file.path(output_dir, "scenario_key.csv"), row.names = FALSE)
write.csv(scenario_summary, file.path(output_dir, "scenario_summary.csv"), row.names = FALSE)

# Scenario truth and audit-support files
write.csv(copying_pairs_truth, file.path(output_dir, "copying_pairs_truth.csv"), row.names = FALSE)
write.csv(answer_changes_long, file.path(output_dir, "answer_changes_long.csv"), row.names = FALSE)

# Optional inspection files
write.csv(response_long, file.path(output_dir, "response_long.csv"), row.names = FALSE)
write.csv(group_check, file.path(output_dir, "group_check.csv"), row.names = FALSE)
write.csv(answer_change_summary, file.path(output_dir, "answer_change_summary.csv"), row.names = FALSE)

# -----------------------------
# 20. Print quick checks
# -----------------------------

cat("\n============================\n")
cat("Scenario counts\n")
cat("============================\n")
print(table(examinee_metadata$true_group))

cat("\n============================\n")
cat("Group-level quick check\n")
cat("============================\n")
print(group_check)

cat("\n============================\n")
cat("Exposed / compromised items\n")
cat("============================\n")
print(compromised_items)

cat("\n============================\n")
cat("Answer-change summary by group\n")
cat("============================\n")
print(aggregate(
  cbind(changed, wrong_to_right, right_to_wrong) ~ true_group,
  data = answer_change_summary,
  FUN = mean
))

cat("\n============================\n")
cat("Files saved to:\n")
cat(output_dir, "\n")
cat("============================\n")
