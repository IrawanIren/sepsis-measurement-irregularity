-- ICU stays with admission and discharge times.
--
-- Source: the derived table icustay_detail of the MIMIC-IV concepts
-- (MIT-LCP/mimic-code, demographics/icustay_detail.sql), built in MySQL as
-- in [15]. The notebook uses subject_id, hadm_id, stay_id, admission_age,
-- icu_intime, icu_outtime, and first_icu_stay.
--
-- Save the result as icustay_detail.csv in the BASE folder.

SELECT *
FROM icustay_detail;
