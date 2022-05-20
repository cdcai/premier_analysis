# Databricks notebook source
#
%run refactored_feature_extraction.py

# COMMAND ----------

#
%run refactored_feature_tokenization.py

# COMMAND ----------

#
%run sequence_trimming.py --out_dir='/dbfs/home/tnk6/premier_output/' --data_dir='/dbfs/home/tnk6/premier/'

# COMMAND ----------

# MAGIC  %pip install openpyxl

# COMMAND ----------

dbutils.fs.mkdirs('/home/tnk6/model_checkpoints/')

# COMMAND ----------

#
%run baseline_models.py --out_dir='/dbfs/home/tnk6/premier_output/' --data_dir='/dbfs/home/tnk6/premier/' --outcome='misa_pt'

# COMMAND ----------

# MAGIC  %pip install keras-tuner

# COMMAND ----------

#
%run model.py --outcome='misa_pt' --day_one --model='dan' --out_dir='/dbfs/home/tnk6/premier_output/' --data_dir='/dbfs/home/tnk6/premier/'

# COMMAND ----------

#
%run model.py --outcome='misa_pt' --day_one --model='hp_dan' --out_dir='/dbfs/home/tnk6/premier_output/' --data_dir='/dbfs/home/tnk6/premier/'

# COMMAND ----------


