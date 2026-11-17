-- Time-stamped urine output for every ICU stay.
--
-- Source: the derived table urine_output of the MIMIC-IV concepts
-- (MIT-LCP/mimic-code, measurement/urine_output.sql), built in MySQL as in
-- [15]. The notebook sums the records of each clock hour; an hour without
-- a record holds zero, so urine output carries no observation mask.
--
-- Save the result as urine_output.csv in the BASE folder.

SELECT
    u.stay_id,
    u.charttime,
    u.urineoutput
FROM urine_output u;
