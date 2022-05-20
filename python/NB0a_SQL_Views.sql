-- Databricks notebook source
-- MAGIC %python
-- MAGIC # Set up Azure storage connection
-- MAGIC spark.conf.set("fs.azure.account.auth.type.davsynapseanalyticsdev.dfs.core.windows.net", "OAuth")
-- MAGIC spark.conf.set("fs.azure.account.oauth.provider.type.davsynapseanalyticsdev.dfs.core.windows.net", "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider")
-- MAGIC spark.conf.set("fs.azure.account.oauth2.client.id.davsynapseanalyticsdev.dfs.core.windows.net", dbutils.secrets.get(scope="dbs-scope-CDH", key="apps-client-id"))
-- MAGIC spark.conf.set("fs.azure.account.oauth2.client.secret.davsynapseanalyticsdev.dfs.core.windows.net", dbutils.secrets.get(scope="dbs-scope-CDH", key="apps-client-secret"))
-- MAGIC spark.conf.set("fs.azure.account.oauth2.client.endpoint.davsynapseanalyticsdev.dfs.core.windows.net", dbutils.secrets.get(scope="dbs-scope-CDH", key="apps-tenant-id-endpoint"))

-- COMMAND ----------

-- MAGIC %md
-- MAGIC # File: views_to_update.conf
-- MAGIC ```
-- MAGIC vw_covid_id
-- MAGIC vw_covid_hx_id
-- MAGIC vw_covid_bill
-- MAGIC vw_covid_bill_lab
-- MAGIC vw_covid_bill_pharm
-- MAGIC vw_covid_bill_oth
-- MAGIC vw_covid_genlab
-- MAGIC vw_covid_lab_res
-- MAGIC vw_covid_lab_sens
-- MAGIC vw_covid_pat
-- MAGIC vw_covid_pat_all
-- MAGIC vw_covid_additional_paticd_diag
-- MAGIC vw_covid_additional_paticd_proc
-- MAGIC vw_covid_paticd_diag
-- MAGIC vw_covid_paticd_proc
-- MAGIC vw_covid_vitals
-- MAGIC vw_covid_hx_genlab
-- MAGIC vw_covid_hx_lab_res
-- MAGIC vw_covid_hx_lab_sens
-- MAGIC vw_covid_hx_bill
-- MAGIC vw_covid_hx_vitals
-- MAGIC ```

-- COMMAND ----------

DROP TABLE IF EXISTS tnk6_demo.vw_covid_id;

-- COMMAND ----------

-- MAGIC %sql
-- MAGIC -- vw_covid_id.sql
-- MAGIC /*
-- MAGIC COVID visit ID lookup
-- MAGIC Case def:
-- MAGIC - U07.1 or B97.29 as a primary or secondary code
-- MAGIC - Admission date on or later than Jan 2020
-- MAGIC NOTE: This also includes all patient visits, not just the COVID ones.
-- MAGIC You'll need to apply the additional filter on the computed variable covid_visit.
-- MAGIC */
-- MAGIC 
-- MAGIC CREATE TABLE tnk6_demo.vw_covid_id 
-- MAGIC USING DELTA
-- MAGIC LOCATION "abfss://cdh@davsynapseanalyticsdev.dfs.core.windows.net/exploratory/databricks_ml/mitre_premier/views/vw_covid_id"
-- MAGIC AS
-- MAGIC WITH covid_pat_key AS (
-- MAGIC     SELECT a.pat_key, a.medrec_key
-- MAGIC       FROM cdh_premier.patdemo as a
-- MAGIC       INNER JOIN (
-- MAGIC         SELECT DISTINCT pat_key
-- MAGIC         FROM cdh_premier.paticd_diag
-- MAGIC         WHERE icd_code IN ('U07.1')
-- MAGIC         AND icd_pri_sec IN ('P', 'S')
-- MAGIC       ) as b
-- MAGIC       ON a.pat_key = b.pat_key
-- MAGIC       AND a.disc_mon >= 2020204
-- MAGIC   UNION
-- MAGIC     SELECT a.pat_key, a.medrec_key
-- MAGIC     FROM cdh_premier.patdemo as a
-- MAGIC     INNER JOIN (
-- MAGIC       SELECT DISTINCT pat_key
-- MAGIC       FROM cdh_premier.paticd_diag
-- MAGIC       WHERE icd_code IN ('B97.29')
-- MAGIC       AND icd_pri_sec IN ('P', 'S')
-- MAGIC     ) as b
-- MAGIC     ON a.pat_key = b.pat_key
-- MAGIC     AND a.disc_mon IN (2020103, 2020204)
-- MAGIC     AND a.adm_mon IN (2020102, 2020103)
-- MAGIC )
-- MAGIC 
-- MAGIC 
-- MAGIC SELECT DISTINCT 
-- MAGIC a.medrec_key, a.pat_key, a.pat_type, a.i_o_ind,
-- MAGIC a.disc_mon, a.disc_mon_seq, d.days_from_index, a.los,
-- MAGIC IF (c.pat_key IS NULL, 0, 1) as covid_visit
-- MAGIC FROM cdh_premier.patdemo as a
-- MAGIC INNER JOIN covid_pat_key as b
-- MAGIC ON a.medrec_key = b.medrec_key
-- MAGIC LEFT JOIN covid_pat_key as c
-- MAGIC ON a.pat_key = c.pat_key
-- MAGIC LEFT JOIN cdh_premier.readmit as d
-- MAGIC ON a.pat_key = d.pat_key
-- MAGIC ORDER BY a.disc_mon, a.disc_mon_seq, a.medrec_key, a.pat_key

-- COMMAND ----------

--vw_covid_hx_id
/*
ID Lookup for all visits prior to latest COVID inpatient visit
NOTE: This should serve as the compliment dataset for the vw_covid_id view
if filtering to inpatient covid visits.
*/
CREATE TABLE tnk6_demo.vw_covid_hx_id 
AS
WITH
    medrec_latest
    AS
    (
        SELECT DISTINCT medrec_key, max(days_from_index) as latest
        FROM tnk6_demo.vw_covid_id
        WHERE i_o_ind = 'I'
            AND covid_visit = 1
        GROUP BY medrec_key
    )

SELECT a.*
FROM tnk6_demo.vw_covid_id as a
    INNER JOIN medrec_latest as b
    ON a.medrec_key = b.medrec_key
        AND a.days_from_index < b.latest
WHERE (i_o_ind != 'I' AND covid_visit = 1) OR covid_visit = 0

-- COMMAND ----------

--vw_covid_bill
CREATE TABLE tnk6_demo.vw_covid_bill 
-- note: Databricks Runtime 8.0, Delta Lake is the default format 
AS
SELECT DISTINCT
    a.pat_key, a.std_chg_code, d.hosp_chg_desc, c.std_dept_desc, c.clin_sum_code,
    c.clin_sum_desc, c.clin_dtl_code, c.clin_dtl_desc, c.std_chg_desc,
    a.serv_day, a.hosp_qty, a.std_qty,
    a.bill_charges, a.bill_cost, a.bill_var_cost, a.bill_fix_cost,
    c.prod_cat_desc, c.prod_class_desc, c.prod_name_desc, c.prod_name_meth_desc
FROM cdh_premier.patbill as a
    INNER JOIN tnk6_demo.vw_covid_id as b
    ON a.pat_key = b.pat_key
    LEFT JOIN cdh_premier.chgmstr as c
    ON a.std_chg_code = c.std_chg_code
    LEFT JOIN cdh_premier.hospchg as d
    ON a.hosp_chg_id = d.hosp_chg_id
WHERE b.covid_visit = 1
    AND b.i_o_ind = 'I';

-- COMMAND ----------

--vw_covid_bill_lab

CREATE TABLE tnk6_demo.vw_covid_bill_lab 
AS
SELECT *
FROM tnk6_demo.vw_covid_bill
WHERE std_dept_desc = 'LABORATORY'

-- COMMAND ----------

--vw_covid_bill_pharm


CREATE TABLE tnk6_demo.vw_covid_bill_pharm 
AS
SELECT *
FROM tnk6_demo.vw_covid_bill
WHERE std_dept_desc = 'PHARMACY'

-- COMMAND ----------

--vw_covid_bill_oth

CREATE TABLE tnk6_demo.vw_covid_bill_oth
AS
SELECT *
FROM tnk6_demo.vw_covid_bill
WHERE std_dept_desc NOT IN ('LABORATORY', 'PHARMACY')

-- COMMAND ----------

--vw_covid_genlab

CREATE TABLE tnk6_demo.vw_covid_genlab 
AS
SELECT a.*
from cdh_premier.genlab as a
    INNER JOIN tnk6_demo.vw_covid_id as b
    ON a.pat_key = b.pat_key
WHERE b.covid_visit = 1
    AND b.i_o_ind = 'I';

-- COMMAND ----------

--vw_covid_lab_res

/*
Gen lab results for all visits previous to latest inpatient COVID visit
*/
CREATE TABLE tnk6_demo.vw_covid_hx_lab_res 

AS
SELECT a.*
from cdh_premier.lab_res as a
    INNER JOIN tnk6_demo.vw_covid_hx_id as b
    ON a.pat_key = b.pat_key

-- COMMAND ----------

--vw_covid_lab_sens
CREATE TABLE tnk6_demo.vw_covid_hx_lab_sens 
AS
SELECT a.*
from cdh_premier.lab_sens as a
    INNER JOIN tnk6_demo.vw_covid_hx_id as b
    ON a.pat_key = b.pat_key

-- COMMAND ----------

--vw_covid_pat
/*
COVID-19 Case dataset (inpatient visits)
*/

CREATE TABLE tnk6_demo.vw_covid_pat 
AS
SELECT
    a.pat_key, a.medrec_key, a.pat_type, b.prov_id, b.i_o_ind, b.race, b.age, b.hispanic_ind, b.gender,
    b.los, e.disc_status_desc, c.adm_type_desc, b.point_of_origin,
    g.std_payor_desc, b.adm_mon, a.disc_mon, a.disc_mon_seq, f.days_from_prior, f.days_from_index
FROM tnk6_demo.vw_covid_id as a
    LEFT JOIN cdh_premier.patdemo as b
    ON a.pat_key = b.pat_key
    LEFT JOIN cdh_premier.admtype as c
    ON b.adm_type = c.adm_type
    LEFT JOIN cdh_premier.disstat as e
    ON b.disc_status = e.disc_status
    LEFT JOIN cdh_premier.readmit as f
    ON a.pat_key = f.pat_key
    LEFT JOIN cdh_premier.payor as g
    ON b.std_payor = g.std_payor
WHERE a.covid_visit = 1
    AND a.i_o_ind = 'I'
ORDER BY a.disc_mon, a.disc_mon_seq, a.medrec_key, a.pat_key;

-- COMMAND ----------

--vw_covid_pat_all
/*
All patient encounters
- COVID Inpatient
- COVID outpatient
- Historical and future inpatient and outpatient encounters matched by medrec to a COVID encounter
*/

CREATE TABLE tnk6_demo.vw_covid_pat_all 
AS
SELECT
    a.pat_key, a.medrec_key, a.pat_type, b.prov_id, a.i_o_ind, a.covid_visit, b.race, b.age, b.hispanic_ind, b.gender,
    b.los, e.disc_status_desc, c.adm_type_desc, b.point_of_origin,
    g.std_payor_desc, b.adm_mon, a.disc_mon, a.disc_mon_seq, f.days_from_prior, f.days_from_index
FROM tnk6_demo.vw_covid_id as a
    LEFT JOIN cdh_premier.patdemo as b
    ON a.pat_key = b.pat_key
    LEFT JOIN cdh_premier.admtype as c
    ON b.adm_type = c.adm_type
    LEFT JOIN cdh_premier.disstat as e
    ON b.disc_status = e.disc_status
    LEFT JOIN cdh_premier.readmit as f
    ON a.pat_key = f.pat_key
    LEFT JOIN cdh_premier.payor as g
    ON b.std_payor = g.std_payor
ORDER BY a.disc_mon, a.disc_mon_seq, a.medrec_key, a.pat_key;

-- COMMAND ----------

--vw_covid_additional_paticd_diag
CREATE TABLE tnk6_demo.vw_covid_additional_paticd_diag 
AS
SELECT DISTINCT
    a.*
FROM cdh_premier.paticd_diag as a
    LEFT JOIN tnk6_demo.vw_covid_id as b
    ON a.pat_key = b.pat_key
WHERE (b.i_o_ind != 'I' AND b.covid_visit = 1) OR b.covid_visit = 0

-- COMMAND ----------

--vw_covid_additional_paticd_proc
CREATE TABLE tnk6_demo.vw_covid_additional_paticd_proc 
AS
SELECT DISTINCT
    a.*
FROM cdh_premier.paticd_proc as a
    LEFT JOIN tnk6_demo.vw_covid_id as b
    ON a.pat_key = b.pat_key
WHERE (b.i_o_ind != 'I' AND b.covid_visit = 1) OR b.covid_visit = 0

-- COMMAND ----------

--vw_covid_paticd_diag
CREATE TABLE tnk6_demo.vw_covid_paticd_diag 
AS
SELECT DISTINCT a.*
FROM cdh_premier.paticd_diag as a
    INNER JOIN tnk6_demo.vw_covid_id as b
    ON a.pat_key = b.pat_key
WHERE b.i_o_ind = 'I'
    AND b.covid_visit = 1;


-- COMMAND ----------

--vw_covid_paticd_proc
CREATE TABLE tnk6_demo.vw_covid_paticd_proc 
AS
SELECT DISTINCT a.*
FROM cdh_premier.paticd_proc as a
    INNER JOIN tnk6_demo.vw_covid_id as b
    ON a.pat_key = b.pat_key
WHERE b.i_o_ind = 'I'
    AND b.covid_visit = 1;

-- COMMAND ----------

--vw_covid_vitals
CREATE TABLE tnk6_demo.vw_covid_vitals 
AS
SELECT a.pat_key,
    a.abnormal_flag,
    a.facility_test_name,
    a.lab_test,
    a.lab_test_loinc_code,
    a.lab_test_result,
    a.lab_test_result_status,
    a.lab_test_result_unit,
    a.numeric_value_operator,
    a.observation_day_number,
    a.observation_time_of_day,
    a.ordering_provider,
    a.result_day_number,
    a.result_time_of_day,
    a.test_result_numeric_value,
    a.year,
    a.quarter
from cdh_premier.vitals as a
    INNER JOIN tnk6_demo.vw_covid_id as b
    ON a.pat_key = b.pat_key
WHERE b.i_o_ind = 'I'
    AND b.covid_visit = 1;

-- COMMAND ----------

--vw_covid_hx_genlab
/*
Gen lab results for all visits previous to latest inpatient COVID visit
*/
CREATE TABLE tnk6_demo.vw_covid_hx_genlab 
AS
SELECT a.*
from cdh_premier.genlab as a
    INNER JOIN tnk6_demo.vw_covid_hx_id as b
    ON a.pat_key = b.pat_key;

-- COMMAND ----------

--vw_covid_hx_lab_res
/*
Gen lab results for all visits previous to latest inpatient COVID visit
*/
CREATE TABLE tnk6_demo.vw_covid_hx_lab_res 
AS
SELECT a.*
from cdh_premier.lab_res as a
    INNER JOIN tnk6_demo.vw_covid_hx_id as b
    ON a.pat_key = b.pat_key

-- COMMAND ----------

--vw_covid_hx_lab_sens
CREATE TABLE tnk6_demo.vw_covid_hx_lab_sens 
AS
SELECT a.*
from cdh_premier.lab_sens as a
    INNER JOIN tnk6_demo.vw_covid_hx_id as b
    ON a.pat_key = b.pat_key


-- COMMAND ----------

--vw_covid_hx_bill
/*
Billing table for all visits previous to latest inpatient COVID visit
*/
CREATE TABLE tnk6_demo.vw_covid_hx_bill 
AS
SELECT DISTINCT
    a.pat_key, a.std_chg_code, d.hosp_chg_desc, c.std_dept_desc, c.clin_sum_code,
    c.clin_sum_desc, c.clin_dtl_code, c.clin_dtl_desc, c.std_chg_desc,
    a.serv_day, a.hosp_qty, a.std_qty,
    a.bill_charges, a.bill_cost, a.bill_var_cost, a.bill_fix_cost,
    c.prod_cat_desc, c.prod_class_desc, c.prod_name_desc, c.prod_name_meth_desc
FROM cdh_premier.patbill as a
    INNER JOIN tnk6_demo.vw_covid_hx_id as b
    ON a.pat_key = b.pat_key
    LEFT JOIN cdh_premier.chgmstr as c
    ON a.std_chg_code = c.std_chg_code
    LEFT JOIN cdh_premier.hospchg as d
    ON a.hosp_chg_id = d.hosp_chg_id;

-- COMMAND ----------

--vw_covid_hx_vitals
CREATE TABLE tnk6_demo.vw_covid_hx_vitals 
AS
SELECT a.pat_key,
    a.abnormal_flag,
    a.facility_test_name,
    a.lab_test,
    a.lab_test_loinc_code,
    a.lab_test_result,
    a.lab_test_result_status,
    a.lab_test_result_unit,
    a.numeric_value_operator,
    a.observation_day_number,
    a.observation_time_of_day,
    a.ordering_provider,
    a.result_day_number,
    a.result_time_of_day,
    a.test_result_numeric_value,
    a.year,
    a.quarter
from cdh_premier.vitals as a
    INNER JOIN tnk6_demo.vw_covid_hx_id as b
    ON a.pat_key = b.pat_key;

-- COMMAND ----------

-- MAGIC %python
-- MAGIC # Testing
-- MAGIC df = spark.sql("select * from tnk6_demo.vw_covid_hx_vitals")
-- MAGIC df = df.toPandas()

-- COMMAND ----------


