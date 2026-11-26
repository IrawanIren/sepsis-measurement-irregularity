-- Hourly SOFA windows with their end times, for the static inputs of the
-- first ICU day.
--
-- Source: the derived table sofa of the MIMIC-IV concepts
-- (MIT-LCP/mimic-code, score/sofa.sql), built in MySQL as in [15]; this is
-- the query sql/hourly_sofa.sql of [15]. Each row is one hourly window of
-- that table, with the 24-hour rolling SOFA score and its six components,
-- the minimum Glasgow Coma Scale, and the vasopressor rates. The windows of
-- that table are not tied to the clock hour of ICU admission, so the
-- notebook (Cell 7b) keeps only the windows whose endtime is at or before
-- the landmark, 24 hours after the clock hour of ICU admission.
--
-- Save the result as hourly_sofa.csv in the BASE folder.

SELECT
    s.stay_id,
    s.hr,
    s.starttime,
    s.endtime,
    s.gcs_min,
    s.rate_epinephrine, s.rate_norepinephrine,
    s.rate_dopamine,    s.rate_dobutamine,
    s.respiration_24hours, s.coagulation_24hours,
    s.liver_24hours,       s.cardiovascular_24hours,
    s.cns_24hours,         s.renal_24hours,
    s.sofa_24hours
FROM sofa s
JOIN icustays ie ON s.stay_id = ie.stay_id
WHERE s.hr BETWEEN 0 AND 23;
