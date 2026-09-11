rm(list = ls())

library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)
library(forecast)

# Configuration -----------------------------------------------------------

data_path <- "data/outputs/plus1_log10_random_shuffled/log_transformed_arima_data.csv"
labels_path <- "data/outputs/plus1_log10_random_shuffled/shuffled_corrected_combined_data.csv"
out_dir <- "data/arimax_results"

dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

target_var <- "plaque_assay"
train_n <- 17
arima_order <- c(1, 0, 0)

rows_to_keep <- c(
  1, 2, 3, 4, 8, 9, 10, 11, 12, 17, 19, 20,
  26, 27, 28, 33, 35, 36, 37, 40, 41, 42, 43
)

# Data preparation --------------------------------------------------------

stationary_ts_df <- read_csv(data_path, show_col_types = FALSE) %>%
  rename_with(~ "time", .cols = 1) %>%
  slice(rows_to_keep) %>%
  arrange(time)

label_df <- read_csv(labels_path, show_col_types = FALSE) %>%
  select(key, granger_label)

all_delvgs <- setdiff(names(stationary_ts_df), c("time", target_var))
delvg_names <- intersect(all_delvgs, label_df$key)

if (length(delvg_names) == 0) {
  stop("No DELVG columns match the label table.")
}

# Rolling one-step-ahead ARIMA baseline ----------------------------------

rolling_arima <- function(y, train_n, order) {
  n <- length(y)
  pred <- rep(NA_real_, n)
  
  if (n <= train_n) {
    return(pred)
  }
  
  for (t in (train_n + 1):n) {
    y_train <- y[1:(t - 1)]
    
    fit <- tryCatch(
      forecast::Arima(
        y = y_train,
        order = order,
        include.mean = TRUE,
        method = "ML"
      ),
      error = function(e) NULL
    )
    
    if (is.null(fit)) {
      next
    }
    
    fc <- tryCatch(
      forecast::forecast(fit, h = 1),
      error = function(e) NULL
    )
    
    if (!is.null(fc)) {
      pred[t] <- as.numeric(fc$mean[1])
    }
  }
  
  pred
}

plaque_df <- stationary_ts_df %>%
  select(time, plaque = all_of(target_var)) %>%
  drop_na() %>%
  arrange(time)

if (nrow(plaque_df) <= train_n) {
  stop("Not enough observations for the selected training window.")
}

plaque_pred <- rolling_arima(
  y = plaque_df$plaque,
  train_n = train_n,
  order = arima_order
)

baseline_test_idx <- (train_n + 1):nrow(plaque_df)
valid_baseline_idx <- baseline_test_idx[
  is.finite(plaque_df$plaque[baseline_test_idx]) &
    is.finite(plaque_pred[baseline_test_idx])
]

if (length(valid_baseline_idx) == 0) {
  stop("The ARIMA baseline produced no valid test predictions.")
}

baseline_mae <- median(
  abs(plaque_df$plaque[valid_baseline_idx] - plaque_pred[valid_baseline_idx])
)

baseline_rmse <- sqrt(mean(
  (plaque_df$plaque[valid_baseline_idx] - plaque_pred[valid_baseline_idx])^2
))

cat(sprintf("ARIMA baseline MAE: %.4f\n", baseline_mae))
cat(sprintf("ARIMA baseline RMSE: %.4f\n", baseline_rmse))

# Rolling one-step-ahead ARIMAX ------------------------------------------

fit_arimax_for_delvg <- function(df, delvg_name, train_n, order) {
  model_df <- df %>%
    select(
      time,
      plaque = all_of(target_var),
      delvg = all_of(delvg_name)
    ) %>%
    drop_na() %>%
    arrange(time) %>%
    mutate(delvg_lag1 = lag(delvg, 1))
  
  n <- nrow(model_df)
  
  if (n <= train_n) {
    return(NULL)
  }
  
  pred <- rep(NA_real_, n)
  
  for (t in (train_n + 1):n) {
    idx_train <- 2:(t - 1)
    
    y_train <- model_df$plaque[idx_train]
    x_train <- model_df$delvg_lag1[idx_train]
    x_new <- model_df$delvg_lag1[t]
    
    if (
      length(y_train) < 5 ||
      any(!is.finite(y_train)) ||
      any(!is.finite(x_train)) ||
      !is.finite(x_new) ||
      sd(x_train) == 0
    ) {
      next
    }
    
    xreg_train <- matrix(
      x_train,
      ncol = 1,
      dimnames = list(NULL, "delvg_lag1")
    )
    
    xreg_new <- matrix(
      x_new,
      nrow = 1,
      dimnames = list(NULL, "delvg_lag1")
    )
    
    fit <- tryCatch(
      forecast::Arima(
        y = y_train,
        order = order,
        xreg = xreg_train,
        include.mean = TRUE,
        method = "ML"
      ),
      error = function(e) NULL
    )
    
    if (is.null(fit)) {
      next
    }
    
    fc <- tryCatch(
      forecast::forecast(fit, h = 1, xreg = xreg_new),
      error = function(e) NULL
    )
    
    if (!is.null(fc)) {
      pred[t] <- as.numeric(fc$mean[1])
    }
  }
  
  test_idx <- (train_n + 1):n
  valid_idx <- test_idx[
    is.finite(model_df$plaque[test_idx]) &
      is.finite(pred[test_idx])
  ]
  
  if (length(valid_idx) == 0) {
    return(NULL)
  }
  
  errors <- model_df$plaque[valid_idx] - pred[valid_idx]
  
  list(
    result = tibble(
      delvg_id = delvg_name,
      n_total = n,
      n_train_initial = train_n,
      n_test_valid = length(valid_idx),
      MAE = median(abs(errors)),
      RMSE = sqrt(mean(errors^2))
    ),
    predictions = tibble(
      delvg_id = delvg_name,
      time = model_df$time[valid_idx],
      observed = model_df$plaque[valid_idx],
      predicted = pred[valid_idx]
    )
  )
}

model_outputs <- lapply(
  delvg_names,
  function(delvg) {
    fit_arimax_for_delvg(
      df = stationary_ts_df,
      delvg_name = delvg,
      train_n = train_n,
      order = arima_order
    )
  }
)

valid_outputs <- Filter(Negate(is.null), model_outputs)

if (length(valid_outputs) == 0) {
  stop("No ARIMAX model produced valid test predictions.")
}

arimax_results <- bind_rows(lapply(valid_outputs, `[[`, "result")) %>%
  left_join(label_df, by = c("delvg_id" = "key"))

arimax_predictions <- bind_rows(
  lapply(valid_outputs, `[[`, "predictions")
)

# Plot 1: observed series and ARIMA baseline -----------------------------

baseline_plot_df <- plaque_df %>%
  mutate(predicted = plaque_pred)

p_arima <- ggplot(baseline_plot_df, aes(x = time)) +
  geom_line(aes(y = plaque, linetype = "Observed"), linewidth = 0.8) +
  geom_line(aes(y = predicted, linetype = "ARIMA forecast"), linewidth = 0.8, na.rm = TRUE) +
  geom_vline(
    xintercept = baseline_plot_df$time[train_n],
    linetype = "dashed",
    linewidth = 0.6
  ) +
  labs(
    title = sprintf("Rolling ARIMA(%d,%d,%d) forecast", arima_order[1], arima_order[2], arima_order[3]),
    subtitle = sprintf("Initial training window: %d observations | One-step-ahead expanding window", train_n),
    x = "Time",
    y = target_var,
    linetype = NULL
  ) +
  theme_minimal(base_size = 12) +
  theme(legend.position = "top")

# Plot 2: ARIMAX predictive performance by Granger label -----------------

label_order <- c("causing", "bi-directional", "caused", "non-related", "shuffled")
present_labels <- unique(na.omit(arimax_results$granger_label))
plot_levels <- c(
  intersect(label_order, present_labels),
  setdiff(present_labels, label_order)
)

plot_results <- arimax_results %>%
  filter(!is.na(granger_label)) %>%
  mutate(
    granger_label = factor(
      granger_label,
      levels = plot_levels
    )
  )

p_arimax_mae <- ggplot(
  plot_results,
  aes(x = granger_label, y = MAE)
) +
  geom_jitter(width = 0.15, height = 0, alpha = 0.75, size = 2) +
  geom_hline(
    yintercept = baseline_mae,
    linetype = "dashed",
    linewidth = 0.7
  ) +
  annotate(
    "text",
    x = Inf,
    y = baseline_mae,
    label = sprintf("ARIMA baseline MAE = %.3f", baseline_mae),
    hjust = 1.05,
    vjust = -0.5,
    size = 3.5
  ) +
  labs(
    title = sprintf("ARIMAX(%d,%d,%d) predictive performance", arima_order[1], arima_order[2], arima_order[3]),
    subtitle = "Lag-1 DELVG as external regressor; rolling one-step-ahead forecasts",
    x = "Granger label",
    y = "Median absolute error (MAE)"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    axis.text.x = element_text(angle = 15, hjust = 1),
    legend.position = "none"
  )

# Objects available for downstream analysis:
# arimax_results, arimax_predictions, baseline_mae, baseline_rmse,
# p_arima, p_arimax_mae

readr::write_csv(arimax_results,     file.path(out_dir, "arimax_results.csv"))
readr::write_csv(arimax_predictions, file.path(out_dir, "arimax_predictions.csv"))

# baseline metrics are only cat()'d — save them too if you want them on disk
readr::write_csv(
  tibble(metric = c("MAE", "RMSE"), value = c(baseline_mae, baseline_rmse)),
  file.path(out_dir, "arima_baseline_metrics.csv")
)

ggsave(file.path(out_dir, "arima_baseline.pdf"),   p_arima,      width = 7, height = 4.5)
ggsave(file.path(out_dir, "arimax_mae.pdf"),       p_arimax_mae, width = 7, height = 4.5)