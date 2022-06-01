# Databricks notebook source
# MAGIC %r
# MAGIC # === Lib
# MAGIC library(dplyr)
# MAGIC library(readr)
# MAGIC library(ggplot2)
# MAGIC library(arrow)
# MAGIC 
# MAGIC data_dir <- file.path("data", "data")
# MAGIC targets_dir <- file.path("data", "targets")
# MAGIC 
# MAGIC # === Read in Data ====================
# MAGIC pat_data <- open_dataset(file.path(data_dir, "vw_covid_id"))
# MAGIC targets <- read.csv(file.path(targets_dir, "icu_targets.csv"), sep = ";")
# MAGIC targets <- as_tibble(targets)
# MAGIC 
# MAGIC # --- Filter pat_data to relevant medrec_keys and pull
# MAGIC pat_filtered <- pat_data %>%
# MAGIC   filter(medrec_key %in% targets[["medrec_key"]]) %>%
# MAGIC   collect()
# MAGIC 
# MAGIC # === re-compute classification
# MAGIC targets <- targets %>%
# MAGIC   group_by(medrec_key) %>%
# MAGIC   mutate(
# MAGIC     status = case_when(
# MAGIC       # Multi-class status:
# MAGIC       # 1 - MISA
# MAGIC       # 2 - Severe COVID (using ICU + COVID ICD as proxy)
# MAGIC       # 3 - Non-severe COVID (still inpatient)
# MAGIC       sum(misa_filled) > 0 ~ 0,
# MAGIC       sum(icu_ever) > 0 ~ 1,
# MAGIC       TRUE ~ 2
# MAGIC     )
# MAGIC   ) %>%
# MAGIC   ungroup()
# MAGIC 
# MAGIC write_csv(targets, file.path(targets_dir, "icu_targets.csv"))
