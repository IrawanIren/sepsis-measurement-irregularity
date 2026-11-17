-- Time-stamped lactate measurements, as extracted in [15].
--
-- Arterial, venous, and mixed specimens are combined. Records are keyed by
-- hospital admission; the notebook maps each admission to the index ICU
-- stay of the cohort and treats zero as a missing result.
--
-- Save the result as lactate.csv in the BASE folder.

SELECT
    l.subject_id,
    l.hadm_id,
    l.charttime,
    l.specimen,
    l.lactate
FROM lactate l;
