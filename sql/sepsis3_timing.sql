-- Sepsis-3 timing and calendar period for every Sepsis-3 ICU stay.
--
-- Used by the notebook (notebooks/PaperA_revision.ipynb, Cell 4) to
-- describe which patients met Sepsis-3 by the 24-hour landmark and to
-- form the temporal split. Same MySQL dialect and table names as the
-- other extraction queries of the repository; the notebook keeps only
-- the stays of the existing cohort, so no cohort filter is needed here.
--
-- Suspicion of infection in the derived sepsis3 table needs both an
-- antibiotic and a culture, and Sepsis-3 adds SOFA >= 2; all three
-- times are exported so that eligibility at the landmark can require
-- every element to have occurred by then. culture_time can be a date
-- without a time (midnight) in MIMIC-IV.
--
-- Save the result as sepsis3_timing.csv in the BASE folder on Drive.

SELECT
    s3.subject_id,
    s3.stay_id,
    ie.hadm_id,
    ie.intime                   AS icu_intime,
    ie.outtime                  AS icu_outtime,
    s3.antibiotic_time,
    s3.culture_time,
    s3.suspected_infection_time,
    s3.sofa_time,
    s3.sofa_score,
    a.deathtime,
    a.dischtime,
    p.dod,
    p.anchor_year_group
FROM sepsis3 s3
JOIN icustays   ie ON s3.stay_id    = ie.stay_id
JOIN admissions a  ON ie.hadm_id    = a.hadm_id
JOIN patients   p  ON ie.subject_id = p.subject_id
WHERE s3.sepsis3 = 1;
