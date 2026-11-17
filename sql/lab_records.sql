-- Time-stamped laboratory results for every ICU stay.
--
-- Source: the pivoted laboratory table of [15] (labevents_pivoted: one row
-- per stay and charttime, one column per variable; see the repository of
-- [15]). NULLs are kept. Some extracts encode a missing result as zero;
-- the notebook treats such records as invalid for the value and the mask
-- alike. Glucose is pooled with the bedside records in the notebook.
-- thrombin, fibrinogen, and albumin feed the static inputs (first valid
-- result in hours 0-23).
--
-- Save the result as pivot_lab.csv in the BASE folder.

SELECT
    l.stay_id,
    l.charttime,
    l.glucose, l.bun, l.creatinine, l.platelets, l.bicarbonate,
    l.aniongap, l.inr, l.pt, l.ptt, l.wbc, l.calcium, l.chloride,
    l.sodium, l.potassium,
    l.albumin, l.fibrinogen, l.thrombin
FROM labevents_pivoted l;
