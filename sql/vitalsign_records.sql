-- Time-stamped vital signs and bedside glucose for every ICU stay.
--
-- Source: the derived table vitalsign of the MIMIC-IV concepts
-- (MIT-LCP/mimic-code, measurement/vitalsign.sql), built in MySQL as in
-- [15]. Every record is kept with its charttime. The notebook keeps the
-- stays of the cohort, assigns each record to a clock hour counted from the
-- clock hour of ICU admission, keeps hours 0-23, and applies the validity
-- rules, so no filtering is done here.
--
-- Save the result as vitalsign.csv in the BASE folder.

SELECT
    v.subject_id,
    v.stay_id,
    v.charttime,
    v.heart_rate, v.sbp, v.dbp, v.mbp,
    v.resp_rate, v.temperature, v.spo2,
    v.glucose
FROM vitalsign v;
