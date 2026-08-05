#!/usr/bin/env Rscript

# Finite-cluster sensitivity analyses for participant-clustered behavioral models.
#
# The analysis reports CR2 standard errors with Satterthwaite degrees of
# freedom and null-imposed Rademacher wild-cluster-bootstrap p values.
# Intercept-only models use the algebraically equivalent participant-cluster
# sign-flip implementation.
#
# Usage:
# Rscript run_behavior_small_sample_inference.R \
#   results/behavior/trial_table_order_carryover.tsv.gz \
#   results/behavior/trialwise_behavior_latents.tsv \
#   derivatives/behavior_small_sample_sensitivity \
#   9999 \
#   20260803

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3L || length(args) > 5L) {
  stop(
    paste(
      "Usage: Rscript run_behavior_small_sample_inference.R",
      "<trial_table_order_carryover.tsv.gz>",
      "<trialwise_behavior_latents.tsv>",
      "<outdir> [B=9999] [seed=20260803]"
    )
  )
}

master_path <- normalizePath(args[[1]], mustWork = TRUE)
latents_path <- normalizePath(args[[2]], mustWork = TRUE)
outdir <- args[[3]]
B <- if (length(args) >= 4L) as.integer(args[[4]]) else 9999L
seed <- if (length(args) >= 5L) as.integer(args[[5]]) else 20260803L
if (!is.finite(B) || B < 999L) stop("B must be at least 999.")
if (!is.finite(seed)) stop("seed must be an integer.")
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

required_packages <- c("clubSandwich", "fwildclusterboot", "dqrng")
missing_packages <- required_packages[!vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing_packages)) {
  stop(
    "Missing R packages: ", paste(missing_packages, collapse = ", "),
    ". Install with: install.packages(c(",
    paste(sprintf("\"%s\"", missing_packages), collapse = ", "), "))"
  )
}

read_tsv_auto <- function(path) {
  connection <- if (grepl("\\.gz$", path, ignore.case = TRUE)) gzfile(path, open = "rt") else file(path, open = "rt")
  on.exit(close(connection), add = TRUE)
  read.delim(
    connection,
    header = TRUE,
    sep = "\t",
    quote = "",
    comment.char = "",
    stringsAsFactors = FALSE,
    check.names = FALSE
  )
}

portable_path <- function(path) {
  resolved <- normalizePath(path, mustWork = TRUE)
  working <- normalizePath(getwd(), mustWork = TRUE)
  prefix <- paste0(working, .Platform$file.sep)
  if (startsWith(resolved, prefix)) {
    return(substring(resolved, nchar(prefix) + 1L))
  }
  basename(resolved)
}

master <- read_tsv_auto(master_path)
latents <- read_tsv_auto(latents_path)

require_columns <- function(data, columns, label) {
  missing <- setdiff(columns, names(data))
  if (length(missing)) {
    stop(label, " is missing columns: ", paste(missing, collapse = ", "))
  }
}

require_columns(
  master,
  c(
    "subject", "condition", "rating_z_subject", "rating_raw",
    "temperature_z_subject", "trial_in_run_c", "run_centered_subject",
    "passive_context_shifted"
  ),
  "Master trial table"
)
require_columns(
  latents,
  c(
    "subject", "condition", "is_reg", "is_up", "prediction_error_z",
    "reg_success_z", "temperature_z_subject", "trial_in_run_c",
    "run_centered_subject", "global_trial_z_subject", "expected_pain_z"
  ),
  "Trialwise latent table"
)

master$subject <- factor(master$subject)
master$is_up <- as.integer(master$condition == "up")
master$is_down <- as.integer(master$condition == "down")
latents$subject <- factor(latents$subject)
latents$is_up <- as.numeric(latents$is_up)
latents$is_reg <- as.numeric(latents$is_reg)
latents$down_success_z <- -as.numeric(latents$prediction_error_z)

# Reproduce the centered state-dependence parameterization used by the archived analysis.
reg_latents <- latents[latents$is_reg == 1 & latents$condition %in% c("up", "down"), , drop = FALSE]
scale_numeric <- function(x) {
  x <- as.numeric(x)
  s <- stats::sd(x, na.rm = TRUE)
  if (!is.finite(s) || s <= 0) return(x - mean(x, na.rm = TRUE))
  as.numeric(scale(x))
}
reg_latents$is_up_c <- reg_latents$is_up - mean(reg_latents$is_up, na.rm = TRUE)
reg_latents$expected_pain_c <- scale_numeric(reg_latents$expected_pain_z)
reg_latents$global_trial_c <- scale_numeric(reg_latents$global_trial_z_subject)

specifications <- list(
  list(
    analysis = "task_validation_standardized_up_vs_passive",
    data = master,
    subset = rep(TRUE, nrow(master)),
    formula = rating_z_subject ~ is_up + is_down + temperature_z_subject + trial_in_run_c + run_centered_subject + passive_context_shifted + factor(subject),
    term = "is_up"
  ),
  list(
    analysis = "task_validation_standardized_down_vs_passive",
    data = master,
    subset = rep(TRUE, nrow(master)),
    formula = rating_z_subject ~ is_up + is_down + temperature_z_subject + trial_in_run_c + run_centered_subject + passive_context_shifted + factor(subject),
    term = "is_down"
  ),
  list(
    analysis = "task_validation_raw_up_vs_passive",
    data = master,
    subset = rep(TRUE, nrow(master)),
    formula = rating_raw ~ is_up + is_down + temperature_z_subject + trial_in_run_c + run_centered_subject + passive_context_shifted + factor(subject),
    term = "is_up"
  ),
  list(
    analysis = "task_validation_raw_down_vs_passive",
    data = master,
    subset = rep(TRUE, nrow(master)),
    formula = rating_raw ~ is_up + is_down + temperature_z_subject + trial_in_run_c + run_centered_subject + passive_context_shifted + factor(subject),
    term = "is_down"
  ),
  list(
    analysis = "mean_direction_corrected_success",
    data = latents,
    subset = latents$is_reg == 1,
    formula = reg_success_z ~ 1,
    term = "(Intercept)"
  ),
  list(
    analysis = "up_regulation_success",
    data = latents,
    subset = latents$condition == "up",
    formula = prediction_error_z ~ 1,
    term = "(Intercept)"
  ),
  list(
    analysis = "down_regulation_success",
    data = latents,
    subset = latents$condition == "down",
    formula = down_success_z ~ 1,
    term = "(Intercept)"
  ),
  list(
    analysis = "up_minus_down_success_asymmetry",
    data = latents,
    subset = latents$is_reg == 1,
    formula = reg_success_z ~ is_up + temperature_z_subject + trial_in_run_c + run_centered_subject + factor(subject),
    term = "is_up"
  ),
  list(
    analysis = "passive_run_position_slope",
    data = master,
    subset = master$condition == "passive",
    formula = rating_z_subject ~ temperature_z_subject + run_centered_subject + I(run_centered_subject^2) + trial_in_run_c + passive_context_shifted + factor(subject),
    term = "run_centered_subject"
  ),
  list(
    analysis = "state_dependence_direction_interaction",
    data = reg_latents,
    subset = rep(TRUE, nrow(reg_latents)),
    formula = reg_success_z ~ is_up_c + expected_pain_c + is_up_c:expected_pain_c + global_trial_c + factor(subject),
    term = "is_up_c:expected_pain_c"
  )
)

extract_first <- function(frame, candidates, default = NA_real_) {
  found <- intersect(candidates, names(frame))
  if (!length(found)) return(default)
  as.numeric(frame[[found[[1]]]][[1]])
}

intercept_wild_cluster <- function(fit, clusters, B, seed_value) {
  # For an intercept-only model, the null-imposed Rademacher
  # wild-cluster-bootstrap t statistic is computed directly from cluster sums.
  # With equal cluster sizes this is the participant-level sign-flip t test.
  y <- as.numeric(stats::model.response(stats::model.frame(fit)))
  cluster_factor <- droplevels(factor(clusters))
  if (length(y) != length(cluster_factor)) {
    stop("Intercept-only bootstrap: outcome and cluster vectors are misaligned.")
  }

  keep <- is.finite(y) & !is.na(cluster_factor)
  y <- y[keep]
  cluster_factor <- droplevels(cluster_factor[keep])

  cluster_sum <- as.numeric(rowsum(y, cluster_factor, reorder = FALSE))
  cluster_n <- as.numeric(table(cluster_factor))
  G <- length(cluster_sum)
  N <- sum(cluster_n)
  if (G < 2L || N <= G) {
    stop("Intercept-only bootstrap requires at least two non-empty clusters.")
  }

  estimate <- sum(cluster_sum) / N
  observed_scores <- cluster_sum - cluster_n * estimate
  finite_sample_factor <- G / (G - 1)
  observed_se <- sqrt(
    finite_sample_factor * sum(observed_scores^2) / (N^2)
  )
  observed_t <- estimate / observed_se

  set.seed(seed_value)
  dqrng::dqset.seed(seed_value)

  # Process bootstrap draws in bounded-memory chunks.
  chunk_size <- min(5000L, B)
  completed <- 0L
  exceedances <- 0L
  while (completed < B) {
    current <- min(chunk_size, B - completed)
    weights <- matrix(
      sample(c(-1, 1), current * G, replace = TRUE),
      nrow = current,
      ncol = G
    )
    beta_star <- as.vector(weights %*% cluster_sum) / N
    score_star <- weights * matrix(
      cluster_sum,
      nrow = current,
      ncol = G,
      byrow = TRUE
    ) - beta_star %o% cluster_n
    se_star <- sqrt(
      finite_sample_factor * rowSums(score_star^2) / (N^2)
    )
    t_star <- beta_star / se_star
    valid <- is.finite(t_star)
    exceedances <- exceedances + sum(abs(t_star[valid]) >= abs(observed_t))
    completed <- completed + current
  }

  list(
    t_stat = observed_t,
    p_val = (exceedances + 1) / (B + 1),
    conf_int = c(NA_real_, NA_real_),
    method = paste0(
      "null-imposed Rademacher cluster sign-flip t; ",
      "intercept-only implementation; G=", G
    )
  )
}

run_specification <- function(spec, index) {
  model_data <- spec$data[spec$subset %in% TRUE, , drop = FALSE]
  model_data$subject <- droplevels(factor(model_data$subject))

  # Retain the fitted data environment because the bootstrap routine
  # reconstructs cluster labels from the model call.
  model_formula <- spec$formula
  environment(model_formula) <- environment()

  fit <- stats::lm(
    model_formula,
    data = model_data,
    na.action = stats::na.omit,
    model = TRUE,
    x = TRUE,
    y = TRUE
  )
  coefficient_values <- stats::coef(fit)
  aliased_terms <- names(coefficient_values)[is.na(coefficient_values)]
  if (length(aliased_terms)) {
    stop(
      spec$analysis,
      ": rank-deficient model; aliased coefficient(s): ",
      paste(aliased_terms, collapse = ", ")
    )
  }
  if (!(spec$term %in% names(coefficient_values))) {
    stop(spec$analysis, ": coefficient not found: ", spec$term)
  }

  cat(sprintf("[%d/%d] %s\n", index, length(specifications), spec$analysis))

  used_names <- rownames(stats::model.frame(fit))
  row_match <- match(used_names, rownames(model_data))
  if (anyNA(row_match)) stop(spec$analysis, ": could not align cluster labels to model rows.")
  clusters <- droplevels(factor(model_data$subject[row_match]))

  cr2 <- clubSandwich::coef_test(
    fit,
    vcov = "CR2",
    cluster = clusters,
    test = "Satterthwaite",
    coefs = spec$term
  )
  cr2_frame <- as.data.frame(cr2)

  if (identical(spec$term, "(Intercept)") && length(coefficient_values) == 1L) {
    boot <- intercept_wild_cluster(
      fit = fit,
      clusters = clusters,
      B = B,
      seed_value = seed + index
    )
    bootstrap_label <- boot$method
  } else {
    set.seed(seed + index)
    dqrng::dqset.seed(seed + index)
    boot <- fwildclusterboot::boottest(
      fit,
      param = spec$term,
      B = B,
      clustid = "subject",
      type = "rademacher",
      impose_null = TRUE,
      bootstrap_type = "fnw11",
      p_val_type = "two-tailed",
      conf_int = FALSE,
      engine = "R-lean"
    )
    bootstrap_label <- "null-imposed fnw11, Rademacher, R-lean"
  }
  boot_ci <- if (is.null(boot$conf_int) || length(boot$conf_int) < 2L) {
    c(NA_real_, NA_real_)
  } else {
    as.numeric(boot$conf_int)
  }

  data.frame(
    analysis = spec$analysis,
    formula = paste(deparse(model_formula), collapse = " "),
    term = spec$term,
    estimate = as.numeric(stats::coef(fit)[[spec$term]]),
    n_observations = stats::nobs(fit),
    n_clusters = nlevels(clusters),
    cr2_se = extract_first(cr2_frame, c("SE", "Std. Error")),
    cr2_t = extract_first(cr2_frame, c("tstat", "t-stat", "t")),
    cr2_df_satterthwaite = extract_first(cr2_frame, c("df_Satt", "d.f.", "df")),
    cr2_p = extract_first(cr2_frame, c("p_Satt", "p-val (Satt)", "p.value", "p")),
    wild_cluster_bootstrap_type = bootstrap_label,
    wild_cluster_B = B,
    wild_cluster_t = as.numeric(boot$t_stat),
    wild_cluster_p = as.numeric(boot$p_val),
    wild_cluster_ci_low = boot_ci[[1]],
    wild_cluster_ci_high = boot_ci[[2]],
    stringsAsFactors = FALSE
  )
}

results <- do.call(
  rbind,
  lapply(seq_along(specifications), function(i) run_specification(specifications[[i]], i))
)

output_path <- file.path(outdir, "behavior_small_sample_inference.tsv")
write.table(results, output_path, sep = "\t", row.names = FALSE, quote = FALSE, na = "")

session_path <- file.path(outdir, "r_session_info.txt")
sink(session_path)
cat("Master input:", portable_path(master_path), "\n")
cat("Latents input:", portable_path(latents_path), "\n")
cat("Bootstrap replications:", B, "\n")
cat("Seed:", seed, "\n\n")
print(sessionInfo())
sink()

cat("Wrote:", output_path, "\n")
cat("Wrote:", session_path, "\n")
