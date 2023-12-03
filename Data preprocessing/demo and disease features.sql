# 3 types of features: demographic(dem_), disease(di_) and time-series lab variables(lab_)
# target: prescription medicine(med_)

#############################
## 1. Demographic Features ##
#############################
# Partial demographic features
CREATE OR REPLACE TABLE `bd4h-mimiciii-project.preprocessing.demo_temp1` AS 
SELECT SUBJECT_ID, 
       DOB, 
       ADMITTIME, 
       HADM_ID, 
       GENDER AS demo_gender, 
       LANGUAGE AS demo_language, 
       RELIGION AS demo_religion, 
       MARITAL_STATUS AS demo_marital_status, 
       ETHNICITY AS demo_ethnicity, 
       AGE AS demo_age
FROM (SELECT p.SUBJECT_ID, 
        p.GENDER, 
        p.DOB, 
        a.ADMITTIME, 
        a.HADM_ID, 
        a.LANGUAGE, 
        a.RELIGION, 
        a.MARITAL_STATUS, 
        a.ETHNICITY, 
        DATE_DIFF(a.ADMITTIME, p.DOB, YEAR) AS AGE
FROM `physionet-data.mimiciii_clinical.admissions` AS a 
INNER JOIN `physionet-data.mimiciii_clinical.patients`  AS p 
ON p.SUBJECT_ID=a.SUBJECT_ID) AS df
WHERE AGE>=18;

# Check primary key: HADM_ID 50796
SELECT cnt,
COUNT(*) AS num
FROM (SELECT HADM_ID,
COUNT(*) AS cnt
FROM `bd4h-mimiciii-project.preprocessing.demo_temp1`
GROUP BY 1) AS df
GROUP BY 1;

# Demographic feature: weight at admission time
-- 226512 Admission Weight (Kg)
-- 226531 Admission Weight (lbs.)
-- 762 Admit Wt
CREATE OR REPLACE TABLE `bd4h-mimiciii-project.preprocessing.demo_temp2` AS 
SELECT HADM_ID,
SUBJECT_ID,
AVG(admission_weight_kg) AS admission_weight_kg
FROM (SELECT HADM_ID,
SUBJECT_ID,
d.itemid, 
CASE WHEN c.itemid = 226531 THEN c.valuenum * 0.45359237 ELSE c.valuenum END AS admission_weight_kg
FROM `physionet-data.mimiciii_clinical.chartevents` c
INNER JOIN `physionet-data.mimiciii_clinical.d_items` d
ON c.itemid=d.itemid
WHERE c.itemid IN (226531, 226512, 762)) AS df
GROUP BY 1,2;

# Check primary key: HADM_ID 48512
SELECT cnt,
COUNT(*) AS num
FROM (SELECT HADM_ID,
COUNT(*) AS cnt
FROM `bd4h-mimiciii-project.preprocessing.demo_temp2`
GROUP BY 1) AS df
GROUP BY 1;

# Demographic feature: weight at admission time
-- 226730 height (cm)
-- 1394 height inches
-- 920 Admit Ht 
CREATE OR REPLACE TABLE `bd4h-mimiciii-project.preprocessing.demo_temp3` AS 
SELECT HADM_ID,
SUBJECT_ID,
AVG(admission_height_cm) AS admission_height_cm
FROM (SELECT HADM_ID,
SUBJECT_ID,
d.itemid, 
CASE WHEN c.itemid IN (1394,920) THEN c.valuenum * 2.54 ELSE c.valuenum END AS admission_height_cm
FROM `physionet-data.mimiciii_clinical.chartevents` c
INNER JOIN `physionet-data.mimiciii_clinical.d_items` d
ON c.itemid=d.itemid
WHERE c.itemid IN (226730, 1394, 920)) AS df
GROUP BY 1,2;

# Check primary key: HADM_ID 38557
SELECT cnt,
COUNT(*) AS num
FROM (SELECT HADM_ID,
COUNT(*) AS cnt
FROM `bd4h-mimiciii-project.preprocessing.demo_temp3`
GROUP BY 1) AS df
GROUP BY 1;

# Final demographic feature table
CREATE OR REPLACE TABLE `bd4h-mimiciii-project.preprocessing.demographic` AS 
SELECT df1.*,
df2.admission_weight_kg AS demo_admission_weight_kg,
df3.admission_height_cm AS demo_admission_height_cm
FROM `bd4h-mimiciii-project.preprocessing.demo_temp1` AS df1
LEFT JOIN `bd4h-mimiciii-project.preprocessing.demo_temp2` AS df2
ON df1.HADM_ID = df2.HADM_ID
LEFT JOIN `bd4h-mimiciii-project.preprocessing.demo_temp3` AS df3
ON df1.HADM_ID = df3.HADM_ID;

# Check primary key: HADM_ID 50796
SELECT cnt,
COUNT(*) AS num
FROM (SELECT HADM_ID,
COUNT(*) AS cnt
FROM `bd4h-mimiciii-project.preprocessing.demographic`
GROUP BY 1) AS df
GROUP BY 1;

##########################
## 2. Disease Features ##
##########################

# To get top 2000 disease -->39:
CREATE OR REPLACE TABLE `bd4h-mimiciii-project.preprocessing.disease` AS
SELECT DISTINCT HADM_ID,
      SUBJECT_ID,
      df1.ICD9_CODE AS di_disease
FROM `physionet-data.mimiciii_clinical.diagnoses_icd` AS df1
INNER JOIN (SELECT ICD9_CODE,
      COUNT(DISTINCT HADM_ID) AS num_admission
FROM `physionet-data.mimiciii_clinical.diagnoses_icd` 
GROUP BY 1 
ORDER BY 2 DESC 
LIMIT 39) AS df2
ON df1.ICD9_CODE = df2.ICD9_CODE;

# Check primary key: HADM_ID,di_disease 625412
SELECT cnt,
COUNT(*) AS num
FROM (SELECT HADM_ID,di_disease,
COUNT(*) AS cnt
FROM `bd4h-mimiciii-project.preprocessing.disease`
GROUP BY 1,2) AS df
GROUP BY 1;

