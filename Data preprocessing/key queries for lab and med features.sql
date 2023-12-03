# cast charttime into dates 
SELECT HADM_ID, SUBJECT_ID, LABEL, valuenum, CAST(CHARTTIME AS DATE) AS chartdate 
FROM `bd4h-mimiciii-project.preprocessing.lab`
ORDER BY SUBJECT_ID, CHARTTIME

# categorize the lab tests to add 
SELECT *, 
      CASE
      WHEN lower(LABEL) like "%blood pressure diastolic%"
      THEN "dbp"
      WHEN lower(LABEL) like "%blood pressure systolic%"
      THEN "sbp"
      WHEN lower(LABEL) like "%glucose%"
      THEN "blood_glucose"
      WHEN LABEL like "%Heart Rate%"
      THEN "hr"
      WHEN LABEL like "%PH (%"
      THEN "PH"
      WHEN LABEL like "%Arterial O2%"
      THEN "bos"
      WHEN LABEL like "%Temperature F%"
      THEN "temp"
      WHEN LABEL like "%Temperature C%"
      THEN "temp"
      END AS LAB,
      CASE
      WHEN LABEL like "%Temperature F%"
      THEN (valuenum-32)/1.8
      ELSE valuenum
      END
FROM `orbital-wording-403922.test.LAB_TEMP1` 
ORDER BY HADM_ID,chartdate

# transform to put labs result on each column output to LAB_TEMP3
SELECT HADM_ID, chartdate, 
  AVG(CASE WHEN LAB='dbp' THEN valuenum END) AS dbp,
  AVG(CASE WHEN LAB='sbp' THEN valuenum END) AS sbp,
  AVG(CASE WHEN LAB='blood_glucose' THEN valuenum END) AS blood_glucose,
  AVG(CASE WHEN LAB='hr' THEN valuenum END) AS hr,  
  AVG(CASE WHEN LAB='PH' THEN valuenum END) AS PH,
  AVG(CASE WHEN LAB='bos' THEN valuenum END) AS bos,
  AVG(CASE WHEN LAB='temp' THEN valuenum END) AS temp,

FROM `orbital-wording-403922.test.LAB_TEMP2` 
GROUP BY HADM_ID, chartdate
ORDER BY HADM_ID


# map different drug names of the same drug to the same
select * from (
select *,
    case 
        when DRUG = 'NS' then 'Sodium Chloride'
        when DRUG like '%Sodium Chloride%' then 'Sodium Chloride'
        when DRUG like '%% Dextrose%' then 'Dextrose'
        when DRUG like '% (%' then split(DRUG, ' (')[safe_ordinal(1)]
    end as curated_drugname
from `physionet-data.mimiciii_clinical.prescriptions`) where curated_drugname is not null

select * from (
select *,
    case 
        when DRUG1 = 'NS' then 'Sodium Chloride'
        when DRUG1 = '1/2 NS' then 'Sodium Chloride'
        when DRUG1 like '%Sodium Chloride%' then 'Sodium Chloride'
        when DRUG1 like '%Dextrose%' then 'Dextrose'
        when DRUG1 like '%D%W%' then 'Dextrose'
        when DRUG1 like '%D%NS' then 'Dextrose'
        when DRUG1 like '%Lidocaine%' then 'Lidocaine'
        when DRUG1 like '%Mannitol%' then 'Mannitol'
        when DRUG1 like '%Lidocaine%' then 'Lidocaine'
        when DRUG1 like '%Miconazole%' then 'Miconazole'
        when DRUG1 like '%Mupirocin%' then 'Mupirocin'
        when DRUG1 like '%Nitroglycerin%' then 'Nitroglycerin'
        when DRUG1 like '%Epidural%' then 'Epidural'
        when DRUG1 like '%Albumin%' then 'Albumin'
        when DRUG1 like '%Bupivacaine%' then 'Bupivacaine'
        when DRUG1 like '%Clobetasol Propionate%' then 'Clobetasol Propionate'
        when DRUG1 like '%Timolol Maleate%' then 'Timolol Maleate'
        when DRUG1 like '%Dorzolamide%' then 'Dorzolamide'
        when DRUG1 like '%Hydrocortisone%' then 'Hydrocortisone'
        when DRUG1 like '%Acetylcysteine%' then 'Acetylcysteine'
        when DRUG1 like '%Acetaminophen%' then 'Acetaminophen'
        when DRUG1 like '%Acyclovir%' then 'Acyclovir'
        when DRUG1 like '%Albuterol%' then 'Albuterol'
        when DRUG1 like '%Alteplase%' then 'Alteplase'
        when DRUG1 like '%Aluminum%' then 'Aluminum Hydroxide'
        when DRUG1 like '%Amiodarone%' then 'Amiodarone'
        when DRUG1 like '%Amitriptyline%' then 'Amitriptyline'
        when DRUG1 like '%Amlodipine%' then 'Amlodipine'
        when DRUG1 like '%Artificial Tear%' then 'Artificial Tear'
        when DRUG1 like '%Aspirin%' then 'Aspirin'
        when DRUG1 like '%Atorvastatin%' then 'Atorvastatin'
        when DRUG1 like '%Atropine Sulfate%' then 'Atropine Sulfate'
        when DRUG1 like '%Bacitracin%' then 'Bacitracin'
        when DRUG1 like '%Beclomethasone%' then 'Beclomethasone Dipropionate'
        when DRUG1 like '%Bupivacaine%' then 'Bupivacaine'
        when DRUG1 like '%Ciprofloxacin%' then 'Ciprofloxacin'
        when DRUG1 like '%Clobetasol Propionate%' then 'Clobetasol Propionate'
        when DRUG1 like '%Divalproex%' then 'Divalproex'
        when DRUG1 like '%Docusate%' then 'Docusate'
        when DRUG1 like '%Erythromycin%' then 'Erythromycin'
        when DRUG1 like '%Fentanyl%' then 'Fentanyl'
        when DRUG1 like '%Heparin' then 'Heparin'
        when DRUG1 like '%Hydrocortisone' then 'Hydrocortisone'
        when DRUG1 like '%Insulin' then 'Insulin'
        when DRUG1 like '%Lansoprazole' then 'Lansoprazole'
        when lower(DRUG1) like '%hydromorphone' then 'Hydromorphone'
        when lower(DRUG1) like '%prednisolone acetate%' then 'Prednisolone Acetate'
        when lower(DRUG1) like '%sodium citrate%' then 'Sodium Citrate'
        when lower(DRUG1) like '%acetazolamide%' then 'Acetazolamide'
        when lower(DRUG1) like '%alprazolam%' then 'Alprazolam'
        when lower(DRUG1) like '%bupropion%' then 'Bupropion'
        when lower(DRUG1) like '%clonidine%' then 'Clonidinen'
        when lower(DRUG1) like '%cyclosporine%' then 'Cyclosporine'
        when lower(DRUG1) like '%dopamine%' then 'Dopamine'
        when lower(DRUG1) like '%epinephrine%' then 'Epinephrine'
	 else DRUG1
    end as curated_drugname
from (select *, regexp_replace(DRUG, r'\(.+\)', "") as DRUG1
from `physionet-data.mimiciii_clinical.prescriptions`)) where curated_drugname is not null

# aggregate one-hot encoded drugs based on data and admission number
SELECT HADM_ID, date,
sum(A01A) over (partition by HADM_ID, date) as l0,
sum(A02A) over (partition by HADM_ID, date) as l1,
sum(A02B) over (partition by HADM_ID, date) as l2,
sum(A03F) over (partition by HADM_ID, date) as l3,
sum(A05A) over (partition by HADM_ID, date) as l4,
sum(A07A) over (partition by HADM_ID, date) as l5,
sum(A07E) over (partition by HADM_ID, date) as l6,
sum(A09A) over (partition by HADM_ID, date) as l7,
sum(A11D) over (partition by HADM_ID, date) as l8,
sum(A11H) over (partition by HADM_ID, date) as l9,
sum(A12B) over (partition by HADM_ID, date) as l10,
sum(A12C) over (partition by HADM_ID, date) as l11,
sum(A16A) over (partition by HADM_ID, date) as l12,
sum(B02A) over (partition by HADM_ID, date) as l13,
sum(B03B) over (partition by HADM_ID, date) as l14,
sum(B05B) over (partition by HADM_ID, date) as l15,
sum(B05C) over (partition by HADM_ID, date) as l16,
sum(C01B) over (partition by HADM_ID, date) as l17,
sum(C01D) over (partition by HADM_ID, date) as l18,
sum(C01E) over (partition by HADM_ID, date) as l19,
sum(C02A) over (partition by HADM_ID, date) as l20,
sum(C05B) over (partition by HADM_ID, date) as l21,
sum(C07A) over (partition by HADM_ID, date) as l22,
sum(C08C) over (partition by HADM_ID, date) as l23,
sum(C08D) over (partition by HADM_ID, date) as l24,
sum(C09A) over (partition by HADM_ID, date) as l25,
sum(C09B) over (partition by HADM_ID, date) as l26,
sum(C09C) over (partition by HADM_ID, date) as l27,
sum(C10A) over (partition by HADM_ID, date) as l28,
sum(D04A) over (partition by HADM_ID, date) as l29,
sum(D06A) over (partition by HADM_ID, date) as l30,
sum(D06B) over (partition by HADM_ID, date) as l31,
sum(D08A) over (partition by HADM_ID, date) as l32,
sum(D10A) over (partition by HADM_ID, date) as l33,
sum(G02C) over (partition by HADM_ID, date) as l34,
sum(G03A) over (partition by HADM_ID, date) as l35,
sum(G03D) over (partition by HADM_ID, date) as l36,
sum(G04C) over (partition by HADM_ID, date) as l37,
sum(H01A) over (partition by HADM_ID, date) as l38,
sum(H02A) over (partition by HADM_ID, date) as l39,
sum(H03B) over (partition by HADM_ID, date) as l40,
sum(H04A) over (partition by HADM_ID, date) as l41,
sum(H05B) over (partition by HADM_ID, date) as l42,
sum(J01A) over (partition by HADM_ID, date) as l43,
sum(J01C) over (partition by HADM_ID, date) as l44,
sum(J01F) over (partition by HADM_ID, date) as l45,
sum(J01G) over (partition by HADM_ID, date) as l46,
sum(J01R) over (partition by HADM_ID, date) as l47,
sum(J02A) over (partition by HADM_ID, date) as l48,
sum(J04B) over (partition by HADM_ID, date) as l49,
sum(J06B) over (partition by HADM_ID, date) as l50,
sum(L01B) over (partition by HADM_ID, date) as l51,
sum(L01D) over (partition by HADM_ID, date) as l52,
sum(L01E) over (partition by HADM_ID, date) as l53,
sum(L02B) over (partition by HADM_ID, date) as l54,
sum(L04A) over (partition by HADM_ID, date) as l55,
sum(M02A) over (partition by HADM_ID, date) as l56,
sum(M03B) over (partition by HADM_ID, date) as l57,
sum(M04A) over (partition by HADM_ID, date) as l58,
sum(M05B) over (partition by HADM_ID, date) as l59,
sum(N01A) over (partition by HADM_ID, date) as l60,
sum(N02B) over (partition by HADM_ID, date) as l61,
sum(N05A) over (partition by HADM_ID, date) as l62,
sum(N05B) over (partition by HADM_ID, date) as l63,
sum(N06A) over (partition by HADM_ID, date) as l64,
sum(N06D) over (partition by HADM_ID, date) as l65,
sum(N07A) over (partition by HADM_ID, date) as l66,
sum(N07B) over (partition by HADM_ID, date) as l67,
sum(P01A) over (partition by HADM_ID, date) as l68,
sum(P01B) over (partition by HADM_ID, date) as l69,
sum(R01B) over (partition by HADM_ID, date) as l70,
sum(R02A) over (partition by HADM_ID, date) as l71,
sum(R03B) over (partition by HADM_ID, date) as l72,
sum(R03C) over (partition by HADM_ID, date) as l73,
sum(R05C) over (partition by HADM_ID, date) as l74,
sum(R05D) over (partition by HADM_ID, date) as l75,
sum(R06A) over (partition by HADM_ID, date) as l76,
sum(S01B) over (partition by HADM_ID, date) as l77,
sum(S01E) over (partition by HADM_ID, date) as l78,
sum(S01F) over (partition by HADM_ID, date) as l79,
sum(S01G) over (partition by HADM_ID, date) as l80,
sum(S01H) over (partition by HADM_ID, date) as l81,
sum(S02A) over (partition by HADM_ID, date) as l82,
sum(S02B) over (partition by HADM_ID, date) as l83,
sum(S02D) over (partition by HADM_ID, date) as l84,
sum(V04C) over (partition by HADM_ID, date) as l85,
sum(V10X) over (partition by HADM_ID, date) as l86
FROM `orbital-wording-403922.test.1129med_1hot` order by HADM_ID, date


