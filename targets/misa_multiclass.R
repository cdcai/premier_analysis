# === Lib
library(dplyr)
library(readr)
library(ggplot2)
library(arrow)

data_dir <- file.path("data", "data")
targets_dir <- file.path("data", "targets")

# === Read in Data ====================
pat_data <- open_dataset(file.path(data_dir, "vw_covid_id"))
targets <- read.csv(file.path(targets_dir, "icu_targets.csv"), sep = ";")
targets <- as_tibble(targets)

# --- Filter pat_data to relevant medrec_keys and pull
pat_filtered <- pat_data %>%
  filter(medrec_key %in% targets[["medrec_key"]]) %>%
  collect()

# === re-compute classification
targets <- targets %>%
  group_by(medrec_key) %>%
  mutate(
    status = case_when(
      # Multi-class status:
      # 1 - MISA
      # 2 - Severe COVID (using ICU + COVID ICD as proxy)
      # 3 - Non-severe COVID (still inpatient)
      sum(misa_filled) > 0 ~ 1,
      sum(icu_ever) > 0 ~ 2,
      TRUE ~ 3
    )
  ) %>%
  ungroup()

write_csv(targets, file.path(targets_dir, "icu_targets.csv"))