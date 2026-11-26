-- Ventilation episodes with their start times, for the static input on
-- invasive ventilation.
--
-- Source: the derived table ventilation of the MIMIC-IV concepts
-- (MIT-LCP/mimic-code, treatment/ventilation.sql), built in MySQL as in
-- [15]; this is the query sql/ventilation.sql of [15]. Episodes that start
-- within 24 hours of ICU admission are kept. The notebook (Cell 7b) counts
-- only invasive episodes that started before the landmark, 24 hours after
-- the clock hour of ICU admission.
--
-- Save the result as ventilation.csv in the BASE folder.

SELECT
    v.stay_id,
    v.starttime,
    v.endtime,
    v.ventilation_status
FROM ventilation v
JOIN icustays ie ON v.stay_id = ie.stay_id
WHERE v.starttime < DATE_ADD(ie.intime, INTERVAL 24 HOUR);
