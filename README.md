# Measurement irregularity in deep learning prediction of sepsis mortality

Code for the paper *Explicit Representations of Measurement Irregularity in
Deep Learning Prediction of Sepsis Mortality: A Controlled Evaluation on
MIMIC-IV*, under review at the International Journal of Intelligent
Engineering and Systems (Paper No. 20267589).

The study asks whether representing the timing of measurement explicitly
improves the prediction, at 24 hours after admission to intensive care, of
30-day mortality in adults with sepsis in MIMIC-IV. The prediction is made
for the patients eligible at that landmark: of the 19,902 patients of an
earlier study, those who had met Sepsis-3 by then and were alive and still in
intensive care. Every input is available at the landmark. Hourly values and
observation masks are rebuilt together from time-stamped records, five input
representations are compared with the partition, architecture, and
hyperparameters of the earlier study held fixed, and GRU-D and a time-aware
attention network serve as comparators. The main contrast is repeated in a
later admission period under an analysis plan fixed before training.

> In the code and comments, **[15]** is the earlier study whose cohort and
> model are reused: I. Irawan, S. Nurmaini, R. Passarella, and J. Jasmir,
> "Observation Window Design Outweighs Architectural Complexity in Deep
> Learning for Sepsis Mortality Prediction", *International Journal of
> Intelligent Engineering and Systems*, Vol. 19, No. 11, pp. 745-766, 2026,
> doi: 10.22266/ijies2026.1130.39. Its code is at
> <https://github.com/IrawanIren/sepsis-mortality-24h>.

## Data

MIMIC-IV is available to credentialed researchers through
[PhysioNet](https://physionet.org/content/mimiciv/) under a data use
agreement. This repository contains no patient-level data, no predictions,
and no trained weights; every model is retrained from this code with fixed
seeds. The stay identifiers, outcome labels, and partitions used in the
paper can be shared with credentialed MIMIC-IV users on request.

## Repository layout

```
.
├── notebooks/
│   └── PaperA_revision.ipynb       The analysis, cell by cell (outputs cleared)
├── src/
│   ├── paper_a_rev.py              Reconstruction of values and masks, derived
│   │                               channels, cutoff test, reconciliation,
│   │                               preprocessing, models, training runners,
│   │                               statistics, operating points
│   ├── windows.py                  Window construction of [15]
│   ├── train.py                    CNN-BiGRU architecture and fold training of [15]
│   └── metrics.py                  Metrics of [15]
├── sql/                            Extraction queries (MySQL, as in [15])
│   ├── icustay_detail.sql          ICU stays
│   ├── sepsis3_timing.sql          Sepsis-3 times, landmark eligibility, period
│   ├── vitalsign_records.sql       Vital signs and bedside glucose
│   ├── lab_records.sql             Laboratory results
│   ├── lactate_records.sql         Lactate
│   ├── urine_output_records.sql    Urine output
│   ├── hourly_sofa_records.sql     Hourly SOFA windows (static inputs)
│   └── ventilation_records.sql     Ventilation episodes (static input)
├── config/
│   ├── hyperparameters_reference.json   Settings of [15] (Table 5)
│   ├── search_space.json                Random search for GRU-D and TBAL
│   ├── selected_hp_GRUD.json            Selected GRU-D setting (Table 6)
│   ├── selected_hp_TBAL.json            Selected TBAL setting (Table 6)
│   ├── analysis_plan_temporal.json      Copy of the first plan
│   ├── analysis_plan_temporal.sha256    Its hash, recorded before its runs
│   ├── analysis_plan_temporal_v2.json   Copy of the second plan, read by Cell 16
│   └── analysis_plan_temporal_v2.sha256 Its hash, recorded before its runs
├── plan/
│   ├── analysis_plan_temporal.json      First temporal validation plan,
│   │                                    committed before its runs
│   └── analysis_plan_temporal_v2.json   Second plan (second revision),
│                                        committed before its runs
├── requirements.txt
└── LICENSE
```

## Requirements

The analysis was run in Google Colab (TensorFlow 2.21.0, Keras 3.13.2), on a
GPU runtime or, when none was available, on a CPU runtime; the run log records
the device of every run. Each model trains in a separate short-lived process,
so the memory of a standard runtime is enough.

```bash
pip install -r requirements.txt
```

## Preparing the data

All files go into one folder, `BASE` (in Colab, a folder on Google Drive).

1. **MIMIC-IV and its derived tables.** Obtain credentialed access to
   MIMIC-IV, load it into MySQL, and build the derived concept tables of
   [MIT-LCP/mimic-code](https://github.com/MIT-LCP/mimic-code) as described
   in the repository of [15].

2. **Cohort, outcome, partition, and settings of [15].** Run the pipeline of
   [15]. The notebook reads its outputs:

   | File in `BASE` | Content |
   |---|---|
   | `sepsis_rebuilt_24h.parquet` | Cohort table of [15]: one row per patient and hour, outcome label (death within 30 days of ICU admission), static inputs |
   | `feature_config_rebuilt.json` | Variable lists of [15] |
   | `cv_folds.csv`, `test_set_ids.csv` | Partition of [15]: five folds and the held-out test set |
   | `best_hp_rebuilt.json`, `best_window_rebuilt.json` | Hyperparameters and window of [15] (values in `config/hyperparameters_reference.json`) |
   | `cache_rebuilt/arrays_h24.npz` | Model arrays of [15] |
   | `results_rebuilt/e1_window_grid.csv` | Needed only by the optional check of Cell 10 |
   | `cache_paperA_v2/mask.npz` | Optional: the mask of the first submission, recovered separately from the source records; without it, R3 on the earlier inputs and the comparison of the two masks are skipped |

3. **Time-stamped records.** Run each query in `sql/` and save the result in
   `BASE` under the name given at the top of the query:

   | Query | File in `BASE` |
   |---|---|
   | `icustay_detail.sql` | `icustay_detail.csv` |
   | `sepsis3_timing.sql` | `sepsis3_timing.csv` |
   | `vitalsign_records.sql` | `vitalsign.csv` |
   | `lab_records.sql` | `pivot_lab.csv` |
   | `lactate_records.sql` | `lactate.csv` |
   | `urine_output_records.sql` | `urine_output.csv` |
   | `hourly_sofa_records.sql` | `hourly_sofa.csv` (or the whole hourly table as `sofa_score.csv`) |
   | `ventilation_records.sql` | `ventilation.csv` |

   The extracts may cover the whole database: the notebook keeps the stays
   of the cohort and hours 0 to 23 itself.

4. **Code.** Cell 2 copies the modules of `src/` from this repository to
   `BASE/src`, so that every run uses the published code, and adds any file of
   `config/` missing from `BASE/config`. Without access to the repository,
   copy `src/` and `config/` into `BASE` by hand.

## Running the analysis

Open `notebooks/PaperA_revision.ipynb` in Colab, set `BASE` in Cell 1, and run
the cells in order. Every finished fold and run is saved at once, so a
session that stops resumes where it left off.

| Cell | What it does |
|---|---|
| 1 | Session setup, paths, and seeds (42, 1, 2) |
| 2 | Copies the modules and any missing configuration file from this repository |
| 3 | Loads the modules, the inputs of [15], and the ICU stays |
| 4 | Index stay of each patient and eligibility at the 24-hour landmark, which defines the analysis cohort |
| 5 | Rebuilds the hourly values and masks from the time-stamped records and writes the training table (once) |
| 6 | Reconciliation with the cohort table of [15] |
| 7 | Derived channels and the cutoff test |
| 7b | Static inputs as of the landmark |
| 8 | Partitions of the eligible patients and the internal early-stopping splits |
| 9 | Audits: winsorisation, static inputs, and the timing of every predictor |
| 10 | Optional: confirms that the environment reproduces the training of [15] |
| 11 | Model definitions (no training) |
| 12 | Trains R1 to R5, T, C, and S |
| 13 | Hyperparameters of GRU-D and TBAL (kept from the first revision; optional new search) |
| 14 | Trains the GRU-D family and TBAL |
| 15 | Earlier inputs |
| 16 | Temporal validation: writes the second analysis plan (or uses the one in `config/`) and prints its SHA-256 before any temporal model is trained |
| 17 | Temporal validation runs |
| 18 | Performance with intervals |
| 19 | Paired comparisons and bounded differences |
| 19b | Exploratory, outside the plan: the later period against values alone (R1) |
| 20 | Operating points, net benefit, and decision curves |
| 21 | Measurement counts by outcome and observation frequency |
| 22 | Figures |
| 23 | Exports aggregate results and the code (no patient-level files) |

## Protocol

* **Cohort, outcome, and partition.** Of the 19,902 patients of [15], the
  analysis keeps those eligible at the 24-hour landmark (Cell 4): the
  antibiotic, the culture, and SOFA >= 2 had all occurred by the landmark, and
  the patient was alive and still in intensive care. The outcome is death
  within 30 days of ICU admission. Every eligible patient keeps the partition
  of [15], its held-out test set or one of its five folds; the other patients
  are neither trained on nor evaluated.
* **Static inputs** must be available at the landmark (Cell 7b). The Charlson
  comorbidity index, whose codes are assigned at hospital discharge, and the
  SOFA score at Sepsis-3 onset are not used. Invasive ventilation, the
  first-day SOFA score and its six components, the minimum Glasgow Coma Scale,
  and vasopressor use are recomputed from records and hourly windows that end
  at or before the landmark, with the definition that best reproduces the
  values of [15] under its own time rule. Every model receives the same 22
  static inputs.
* **Inputs.** Each record is assigned to a clock hour counted from the clock
  hour of ICU admission, and only hours 0 to 23 are used. The value of an hour
  is the mean of its valid records, and the mask is one when the hour holds
  at least one. Values are carried forward; nothing is filled backward, and
  hours before the first measurement take the median of the training
  patients. The cutoff test checks that no channel depends on later hours.
* **Prediction time.** One landmark at the end of hour 23. Training uses
  six-hour windows with a stride of one hour; each patient is scored once, on
  hours 18 to 23.
* **Early stopping** monitors the AUROC of an internal 15% of the training
  patients, stratified by outcome (`random_state` 100 + k for fold k, and 7 for
  the test and temporal protocols), with patience 4 and at most 40 epochs; the
  weights of the best epoch are kept. The evaluated patients take no part in
  training, early stopping, or any fitted preprocessing.
* **Calibration.** Platt scaling is fitted on the same internal 15% in the
  test and temporal protocols.
* **Seeds** 42, 1, and 2; probabilities are averaged across seeds before any
  metric is computed.
* **Hyperparameters.** The settings of [15] are used without retuning for the
  channel-based models. GRU-D and TBAL use the settings selected in the first
  revision by the random search in `config/search_space.json`, which used
  development folds 1 to 3 only.
* **Statistics.** DeLong tests for the AUROC; the AUPRC is the average
  precision without interpolation, with 2,000 paired bootstrap resamples
  (seed 2026); Benjamini-Hochberg adjustment within each family of contrasts.
  A difference is bounded within ±0.01 when its 90% DeLong interval lies
  inside the margin.
* **Temporal validation.** Models are trained on eligible patients admitted
  from 2008 to 2016 and evaluated on eligible patients admitted from 2017 to
  2019, under the second plan, `plan/analysis_plan_temporal_v2.json`. It keeps
  the models, seeds, contrasts, margin, operating points, and bootstrap
  settings of the first plan and records that the period 2017 to 2019 was
  evaluated once under it. The plan was written on 24 November 2026 at
  16:44:54 UTC, when its SHA-256 was recorded
  (`config/analysis_plan_temporal_v2.sha256`), and committed to this
  repository at 16:51:49 UTC (commit
  [f4bf8c4](https://github.com/IrawanIren/sepsis-measurement-irregularity/commit/f4bf8c4)),
  before any model of the second revision was trained. The copy in `config/`
  is identical and is read by Cell 16. Cell 19b compares the alternatives with
  values alone (R1) in the later period; these contrasts are not in the plan
  and are reported as exploratory.
* **First plan.** In the first revision, all 19,902 patients with the static
  inputs of [15] were evaluated under `plan/analysis_plan_temporal.json`. That
  plan was written on 17 November 2026 at 00:47:18 UTC, when its SHA-256 was
  recorded (`config/analysis_plan_temporal.sha256`), and committed to this
  repository at 01:10:05 UTC (commit
  [33edf24](https://github.com/IrawanIren/sepsis-measurement-irregularity/commit/33edf24)),
  before the temporal runs; these finished at 02:30:38 UTC. The copy in
  `config/` is identical and was added later with the rest of the code.

Every model, the reference R2 included, is trained by the notebook on the
same partitions, so all comparisons rest on one training procedure.

## Outputs

The notebook writes `results_paperA_rev2/`, `tables_paperA_rev2/`, and
`figures_paperA_rev2/` in `BASE`. The training table, caches, and predictions
contain patient-level data and stay in `BASE`; Cell 23 exports aggregate files
only. Aggregate result tables and figures will be added to this repository
upon publication.

## Versions

* **Second revision** (this version): eligible patients only, static inputs
  as of the landmark, and the second temporal plan.
* **First revision**: commit
  [f68f489](https://github.com/IrawanIren/sepsis-measurement-irregularity/commit/f68f489),
  with all 19,902 patients of [15] and the static inputs of [15].

## Citation

Citation details will be added upon publication.

## License

MIT License; see `LICENSE`.
