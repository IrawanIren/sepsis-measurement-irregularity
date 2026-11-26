"""Rebuilt hourly inputs, derived channels, models, and statistics.

Code of the revised analysis submitted to IJIES (Paper No. 20267589).
Three rules run through the whole module.

* Values and observation masks are built together from the same
  time-stamped records, with one set of rules for hour assignment,
  validity, and aggregation. A mask of one therefore always marks an hour
  whose value is the mean of the valid records in that hour, and a value
  changes only at such an hour.
* Nothing is filled backward. Hours before the first observation stay
  missing until a statistic of the training patients is supplied, so no
  input of any window can draw on a later measurement.
* Every fitted quantity (fill values, winsorisation limits, scaling,
  Platt scaling) is estimated on the patients used for gradient updates,
  and early stopping monitors an internal split of the training data,
  never the patients being evaluated.

Sections: reading and hour assignment; the hourly grid; derived channels;
landmark eligibility; reconciliation with the cohort table of [15];
fold-wise preprocessing; models; training runners that resume; statistics.
"""

import hashlib
import itertools
import json
import os
import subprocess
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm, rankdata
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:                                    # the data and statistics sections
    import keras                        # run without a deep learning library
    from keras import layers, ops
except ImportError:                     # pragma: no cover
    keras = None

N_HOURS = 24

VITAL = ['heart_rate', 'sbp', 'dbp', 'mbp', 'resp_rate', 'temperature', 'spo2']
LAB = ['glucose', 'bun', 'creatinine', 'platelets', 'bicarbonate', 'aniongap',
       'inr', 'pt', 'ptt', 'wbc', 'calcium', 'chloride', 'sodium', 'potassium']
UNMASKED = ('urineoutput',)
STATIC_LABS = ('thrombin', 'fibrinogen', 'albumin')

# Laboratory extracts encode some missing results as zero. Zero is
# physiologically impossible for these variables, so such a record is
# invalid, for the value and the mask alike.
ZERO_MEANS_MISSING = {
    'platelets', 'wbc', 'aniongap', 'bicarbonate', 'bun', 'calcium',
    'chloride', 'creatinine', 'glucose', 'sodium', 'potassium', 'inr',
    'pt', 'ptt', 'albumin', 'fibrinogen', 'thrombin', 'lactate',
}


# ======================================================================
# Reading and hour assignment
# ======================================================================

def detect_sep(path):
    """Extracts vary in delimiter, so detect it from the header."""
    with open(path) as f:
        header = f.readline()
    return ';' if header.count(';') > header.count(',') else ','


def read_extract(path, key, cols, keep, chunksize=2_000_000):
    """Rows of a time-stamped extract that belong to the cohort.

    Only the key, charttime and the requested columns are read, in
    chunks, so that a whole-database extract fits in memory. Returns the
    frame, the requested columns present, and those absent from the file.
    """
    sep = detect_sep(path)
    header = list(pd.read_csv(path, sep=sep, nrows=0, quotechar='"').columns)
    for c in (key, 'charttime'):
        if c not in header:
            raise KeyError(f"{os.path.basename(path)} has no column {c!r}; "
                           f"its columns are {header}")
    present = [c for c in cols if c in header]
    absent = [c for c in cols if c not in header]
    keep = pd.Index(pd.unique(np.asarray(keep)))
    parts = []
    for chunk in pd.read_csv(path, sep=sep, quotechar='"', chunksize=chunksize,
                             usecols=[key, 'charttime'] + present, low_memory=False):
        chunk = chunk[chunk[key].isin(keep)]
        if len(chunk):
            parts.append(chunk)
    d = (pd.concat(parts, ignore_index=True) if parts
         else pd.DataFrame(columns=[key, 'charttime'] + present))
    d['charttime'] = pd.to_datetime(d['charttime'], errors='coerce')
    for c in present:
        d[c] = pd.to_numeric(d[c], errors='coerce')
    return d, present, absent


def calendar_hour(charttime, intime):
    """Hours from the clock hour of ICU admission, the convention of the
    cohort table of [15]: records at 19:05 and 19:35 share an hour."""
    return (charttime.dt.floor('h') - intime.dt.floor('h')).dt.total_seconds() // 3600


def _bounds(bounds, c):
    lo, hi = bounds.get(c, (-np.inf, np.inf))
    lo = lo if lo is not None and np.isfinite(lo) else -np.inf
    hi = hi if hi is not None and np.isfinite(hi) else np.inf
    return lo, hi


def _valid(v, c, bounds):
    """One validity rule for value and mask: zero standing for a missing
    result is invalid, other values are clipped to the bounds of [15]."""
    v = np.asarray(v, dtype=float)
    if c in ZERO_MEANS_MISSING:
        v = np.where(v == 0, np.nan, v)
    lo, hi = _bounds(bounds, c)
    return np.clip(v, lo, hi)


def accumulate(d, key, cols, key_to_row, key_to_intime, n_rows, bounds,
               n_hours=N_HOURS):
    """Sums and counts of valid records per patient, hour, and variable.

    Records are kept when they fall in hours 0..n_hours-1 of the stay, so
    nothing at or after the landmark (the end of hour 23) can enter.
    """
    row = d[key].map(key_to_row)
    intime = d[key].map(key_to_intime)
    hour = calendar_hour(d['charttime'], intime)
    keep = (row.notna() & hour.between(0, n_hours - 1)).to_numpy()
    landmark = intime.dt.floor('h') + pd.Timedelta(hours=n_hours)
    info = {
        'records read': int(len(d)),
        'records in hours 0-23': int(keep.sum()),
        'records in hour 0 before ICU admission':
            int((keep & (d['charttime'] < intime).to_numpy()).sum()),
        'records at or after the landmark kept':
            int((keep & (d['charttime'] >= landmark).to_numpy()).sum()),
    }
    r = row.to_numpy()[keep].astype(np.int64)
    h = hour.to_numpy()[keep].astype(np.int64)
    flat = r * n_hours + h
    sums, counts = {}, {}
    for c in cols:
        v = _valid(d[c].to_numpy(dtype=float)[keep], c, bounds)
        ok = np.isfinite(v)
        sums[c] = np.bincount(flat[ok], weights=v[ok],
                              minlength=n_rows * n_hours).reshape(n_rows, n_hours)
        counts[c] = np.bincount(flat[ok], minlength=n_rows * n_hours).reshape(n_rows, n_hours)
        info[f'valid records: {c}'] = int(ok.sum())
    return sums, counts, info


def combine(parts, features, n_rows, n_hours=N_HOURS):
    """Hourly mean over all valid records of all sources, the observation
    mask, and the number of records. A variable kept in two sources
    (glucose: laboratory and bedside) pools both, so its value and its
    mask follow the same source rule."""
    S = np.zeros((n_rows, n_hours, len(features)))
    N = np.zeros((n_rows, n_hours, len(features)), dtype=np.int32)
    for sums, counts in parts:
        for j, f in enumerate(features):
            if f in sums:
                S[:, :, j] += sums[f]
                N[:, :, j] += counts[f]
    with np.errstate(invalid='ignore', divide='ignore'):
        V = np.where(N > 0, S / np.maximum(N, 1), np.nan)
    return V.astype(np.float32), (N > 0).astype(np.float32), N


def hourly_total(d, key, col, key_to_row, key_to_intime, n_rows, bounds,
                 n_hours=N_HOURS):
    """Urine output per hour: the sum and the mean of the valid records.
    An hour without a record holds zero (no output), so this variable
    carries no mask and needs no fill."""
    sums, counts, info = accumulate(d, key, [col], key_to_row, key_to_intime,
                                    n_rows, bounds, n_hours)
    s, n = sums[col], counts[col]
    lo, hi = _bounds(bounds, col)
    total = np.clip(s, lo, hi)
    with np.errstate(invalid='ignore', divide='ignore'):
        mean = np.clip(np.where(n > 0, s / np.maximum(n, 1), 0.0), lo, hi)
    return total.astype(np.float32), mean.astype(np.float32), info


def first_values(d, key, cols, key_to_row, key_to_intime, n_rows, bounds,
                 n_hours=N_HOURS):
    """First valid result in hours 0-23 for each patient, and its hour.
    Used for the three nearly constant laboratory variables of the static
    pathway; a patient never tested in the first day is left missing."""
    row = d[key].map(key_to_row)
    intime = d[key].map(key_to_intime)
    hour = calendar_hour(d['charttime'], intime)
    keep = (row.notna() & hour.between(0, n_hours - 1)).to_numpy()
    e = d.loc[keep, ['charttime'] + list(cols)].copy()
    e['row'] = row.to_numpy()[keep].astype(np.int64)
    e['hour'] = hour.to_numpy()[keep].astype(np.int64)
    e = e.sort_values(['row', 'charttime'], kind='mergesort')
    out = {}
    for c in cols:
        v = _valid(e[c].to_numpy(dtype=float), c, bounds)
        ok = np.isfinite(v)
        first = (e.loc[ok, ['row', 'hour']].assign(v=v[ok])
                  .drop_duplicates('row', keep='first'))
        vals = np.full(n_rows, np.nan)
        hrs = np.full(n_rows, np.nan)
        vals[first['row'].to_numpy()] = first['v'].to_numpy()
        hrs[first['row'].to_numpy()] = first['hour'].to_numpy()
        out[c] = (vals, hrs)
    return out


# ======================================================================
# The training table: one CSV row per patient and hour
# ======================================================================

def training_table(pids, stays, y, V, M, U, stat_num, cat_raw, num_names, cat_names,
                   features, urine, eligible, period, split, id_col='subject_id'):
    """The new training data in the long format of the cohort table of
    [15]: per patient and hour, the carried-forward value of each variable
    (empty before its first observation, filled per fold during training),
    its observation mask m_<variable>, urine output, the static inputs, and
    the outcome, landmark eligibility, calendar period, and partition."""
    n, T, J = V.shape
    F = ffill_causal(V)
    cols = {id_col: np.repeat(pids, T), 'stay_id': np.repeat(stays, T),
            'hour': np.tile(np.arange(T), n), 'label': np.repeat(np.asarray(y).astype(int), T),
            'eligible': np.repeat(np.asarray(eligible).astype(int), T),
            'period': np.repeat(period, T), 'split': np.repeat(split, T)}
    for j, f in enumerate(features):
        cols[f] = F[:, :, j].reshape(-1)
        cols[f'm_{f}'] = M[:, :, j].reshape(-1).astype(np.int8)
    cols[urine] = np.asarray(U).reshape(-1)
    for j, c in enumerate(num_names):
        cols[c] = np.repeat(stat_num[:, j], T)
    for j, c in enumerate(cat_names):
        cols[c] = np.repeat(cat_raw[:, j], T)
    return pd.DataFrame(cols)


def read_training_table(path, id_col, pids, features, urine, num_names, n_hours=N_HOURS):
    """Arrays for training, read back from the CSV in the patient order of
    the arrays of [15]. The hourly value V is the carried-forward value
    where the mask is one and missing elsewhere."""
    t = pd.read_csv(path, low_memory=False).sort_values([id_col, 'hour'], kind='mergesort')
    n, T, J = len(pids), n_hours, len(features)
    first = t.iloc[::T]
    assert len(t) == n * T and np.array_equal(first[id_col].to_numpy(), np.asarray(pids)), \
        'the training table does not match the cohort; delete it and rebuild'
    F = t[features].to_numpy(dtype=np.float32).reshape(n, T, J)
    M = t[[f'm_{f}' for f in features]].to_numpy(dtype=np.float32).reshape(n, T, J)
    return {'V': np.where(M == 1, F, np.nan).astype(np.float32), 'M': M,
            'U': t[urine].to_numpy(dtype=np.float32).reshape(n, T),
            'stat_num': first[list(num_names)].to_numpy(dtype=float),
            'stay_id': first['stay_id'].to_numpy(),
            'eligible': first['eligible'].to_numpy().astype(bool),
            'period': first['period'].to_numpy(dtype=float),
            'split': first['split'].astype(str).to_numpy()}


def table_dictionary(features, urine, num_names, cat_names, static_labs, urine_source,
                     id_col='subject_id'):
    """Column descriptions of the training table (aggregate, safe to share)."""
    rows = [(id_col, 'patient identifier (MIMIC-IV)'),
            ('stay_id', 'index ICU stay: the stay behind the row of the patient in [15] (Cell 4)'),
            ('hour', 'hour counted from the clock hour of ICU admission, 0-23'),
            ('label', 'death within 30 days of ICU admission (1) or not (0)'),
            ('eligible', '1 if antibiotic, culture, and SOFA >= 2 had all occurred by the 24-hour landmark '
                         'and the patient was alive and in the ICU then; only these patients are analysed'),
            ('period', 'first year of anchor_year_group (temporal split: < 2017 develop, >= 2017 evaluate)'),
            ('split', 'partition of [15] (fold 1-5 or test)')]
    for f in features:
        rows.append((f, 'mean of the valid records in the hour, carried forward; empty before the '
                        'first observation (filled with the training median per fold)'))
        rows.append((f'm_{f}', '1 if at least one valid record of the variable fell in the hour'))
    rows.append((urine, f'urine output per hour, zero without a record, no mask ({urine_source})'))
    for c in num_names:
        rows.append((c, 'static input of [15]' + ('; first valid result in hours 0-23, empty if never '
                                                  'measured' if c in static_labs else '')))
    for c in cat_names:
        rows.append((c, 'static category of [15], one-hot encoded for training'))
    return pd.DataFrame(rows, columns=['column', 'meaning'])


# ======================================================================
# Derived channels (all causal: hour t uses hours 0..t only)
# ======================================================================

def ffill_causal(V):
    """Last observation carried forward; hours before the first
    observation stay missing (no backward fill)."""
    F = np.array(V, dtype=np.float32, copy=True)
    for t in range(1, F.shape[1]):
        gap = np.isnan(F[:, t])
        F[:, t][gap] = F[:, t - 1][gap]
    return F


def carry_forward_difference(F):
    """Hourly difference of the carried-forward series, as in [15]: zero
    at every hour without a new value and, since nothing is filled
    backward, zero up to and including the first observation."""
    D = np.zeros_like(F)
    D[:, 1:] = F[:, 1:] - F[:, :-1]
    D[np.isnan(D)] = 0.0
    return D


def hourly_difference(U):
    D = np.zeros_like(U)
    D[:, 1:] = U[:, 1:] - U[:, :-1]
    return D


def delta_reset(M, T=N_HOURS):
    """Eq. (2): zero at an observed hour, one more than the previous hour
    otherwise, capped at T, and T before the first observation."""
    out = np.empty(M.shape, dtype=np.float32)
    out[:, 0] = np.where(M[:, 0] == 1, 0.0, T)
    for t in range(1, M.shape[1]):
        out[:, t] = np.where(M[:, t] == 1, 0.0, np.minimum(out[:, t - 1] + 1.0, T))
    return out


def delta_che(M):
    """Gap convention of Che et al. [16] for hourly steps: zero at the
    first step, one after an observed step, one more than before after an
    unobserved step. An observed hour carries the gap since the previous
    observation instead of zero."""
    out = np.zeros(M.shape, dtype=np.float32)
    for t in range(1, M.shape[1]):
        out[:, t] = np.where(M[:, t - 1] == 1, 1.0, out[:, t - 1] + 1.0)
    return out


def change_observed(V, M):
    """Eq. (3): change between consecutive observed values, held until
    the next observation; zero until the second observation."""
    n, T, J = V.shape
    out = np.zeros((n, T, J), dtype=np.float32)
    last = np.full((n, J), np.nan)
    held = np.zeros((n, J), dtype=np.float32)
    for t in range(T):
        seen = M[:, t] == 1
        cur = V[:, t].astype(float)
        both = seen & ~np.isnan(last)
        held = np.where(both, (cur - last).astype(np.float32), held)
        out[:, t] = held
        last = np.where(seen, cur, last)
    return out


def cumulative_counts(M):
    """Number of hours with a measurement up to and including hour t."""
    return np.cumsum(M, axis=1, dtype=np.float32)


def components(V, M, U):
    """Every channel type, in raw units, from the aligned grid."""
    F = ffill_causal(V)
    return {'V': V, 'M': M, 'F': F, 'DX': carry_forward_difference(F),
            'U': U, 'DU': hourly_difference(U), 'DT': delta_reset(M),
            'DTC': delta_che(M), 'D': change_observed(V, M),
            'C': cumulative_counts(M)}


def causality_test(V, M, U, cuts=(5, 11, 17), n=300, seed=0):
    """Change every hour after a cut and confirm that no channel at or
    before the cut changes. A training window that ends at the cut sees
    only those hours, so its inputs cannot depend on later records."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(V), size=min(n, len(V)), replace=False)
    base = components(V[idx], M[idx], U[idx])
    rows = []
    for cut in cuts:
        V2, M2, U2 = V[idx].copy(), M[idx].copy(), U[idx].copy()
        tail = (slice(None), slice(cut + 1, None))
        M2[tail] = rng.integers(0, 2, size=M2[tail].shape)
        V2[tail] = np.where(M2[tail] == 1,
                            rng.normal(100, 30, size=V2[tail].shape), np.nan)
        U2[tail] = rng.uniform(0, 500, size=U2[tail].shape)
        alt = components(V2, M2, U2)
        for k in base:
            same = np.array_equal(base[k][:, :cut + 1], alt[k][:, :cut + 1], equal_nan=True)
            rows.append({'hours compared': f'0-{cut}', 'channel': k,
                         'unchanged after editing later hours': bool(same)})
    return pd.DataFrame(rows)


# ======================================================================
# Landmark eligibility
# ======================================================================

def truthy(x):
    """1/0, True/False and 't'/'f' exports of a boolean column."""
    return pd.Series(x).astype(str).str.strip().str.lower().isin(['1', '1.0', 'true', 't'])


def resolve_stays(pids, table_first, timing, icu, label_col='label'):
    """The ICU stay behind each patient's row in the cohort table of [15].

    The cohort query of [15] keeps every stay with Sepsis-3 that is the
    first ICU stay of its hospital admission and lasts 24 h or more, so a
    patient can have one such stay per admission, and the first ICU stay
    of the patient may belong to another admission. When the table carries
    stay_id it is used as is. Otherwise each patient's candidates are the
    stays meeting those criteria in the timing file; with several, the one
    whose SOFA at onset, age, and 30-day outcome agree with the table is
    taken, and the earliest when that does not decide. A patient without a
    candidate gets the earliest first ICU stay of 24 h or more and, having
    no Sepsis-3 times, cannot be eligible.
    """
    pids = np.asarray(pids)
    tf = table_first.loc[pids]
    if 'stay_id' in tf.columns and tf['stay_id'].notna().all():
        return tf['stay_id'].astype('int64').to_numpy(), pd.DataFrame(
            [{'how the ICU stay was found': 'stay_id column of the cohort table', 'patients': len(pids)}])
    ic = icu.drop_duplicates('stay_id').set_index('stay_id')
    t = timing[timing['subject_id'].isin(pids)].drop_duplicates('stay_id').copy()
    t['icu_intime'] = pd.to_datetime(t['icu_intime'], errors='coerce')
    t['icu_outtime'] = pd.to_datetime(t['icu_outtime'], errors='coerce')
    t['first'] = truthy(t['stay_id'].map(ic['first_icu_stay'])).to_numpy()
    t = t[t['first'] & ((t['icu_outtime'] - t['icu_intime']) >= pd.Timedelta(days=1))].copy()
    subj = t['subject_id']
    t['score'] = 0
    if 'sofa_onset' in tf.columns and 'sofa_score' in t.columns:
        t['score'] += (pd.to_numeric(t['sofa_score'], errors='coerce').to_numpy()
                       == subj.map(tf['sofa_onset']).to_numpy()).astype(int)
    if 'admission_age' in tf.columns and 'admission_age' in ic.columns:
        age = pd.to_numeric(t['stay_id'].map(ic['admission_age']), errors='coerce')
        t['score'] += ((age - subj.map(tf['admission_age'])).abs() <= 1).astype(int).to_numpy()
    if label_col in tf.columns and 'dod' in t.columns:
        dod = pd.to_datetime(t['dod'], errors='coerce')
        lab = (dod.notna() & (dod <= t['icu_intime'] + pd.Timedelta(days=30))).astype(int)
        t['score'] += (lab.to_numpy() == subj.map(tf[label_col]).astype(int).to_numpy()).astype(int)
    t['n'] = t.groupby('subject_id')['stay_id'].transform('size')
    top = t.groupby('subject_id')['score'].transform('max')
    t['n_top'] = (t['score'] == top).groupby(t['subject_id']).transform('sum')
    best = (t.sort_values(['subject_id', 'score', 'icu_intime'], ascending=[True, False, True])
              .drop_duplicates('subject_id').set_index('subject_id'))
    stays = best['stay_id'].reindex(pids)

    # No candidate: earliest first ICU stay of 24 h or more, else earliest first ICU stay.
    lack = stays.isna().to_numpy()
    if lack.any():
        f = icu[icu['subject_id'].isin(pids[lack]) & truthy(icu['first_icu_stay']).to_numpy()].copy()
        f['icu_intime'] = pd.to_datetime(f['icu_intime'], errors='coerce')
        f['long'] = (pd.to_datetime(f['icu_outtime'], errors='coerce') - f['icu_intime']) >= pd.Timedelta(days=1)
        f = f.sort_values(['subject_id', 'long', 'icu_intime'], ascending=[True, False, True])
        stays[lack] = f.drop_duplicates('subject_id').set_index('subject_id')['stay_id'].reindex(pids[lack]).to_numpy()
    b = best.reindex(pids)
    report = pd.DataFrame([
        {'how the ICU stay was found': 'one stay meeting the cohort criteria of [15]',
         'patients': int((b['n'] == 1).sum())},
        {'how the ICU stay was found': 'several such stays: the one agreeing with the table (SOFA at onset, age, outcome)',
         'patients': int(((b['n'] > 1) & (b['n_top'] == 1)).sum())},
        {'how the ICU stay was found': 'several such stays agreeing equally: the earliest',
         'patients': int(((b['n'] > 1) & (b['n_top'] > 1)).sum())},
        {'how the ICU stay was found': 'no such stay in the timing file: earliest first ICU stay (not eligible)',
         'patients': int(lack.sum())}])
    if stays.isna().any():
        raise ValueError(f'{int(stays.isna().sum())} patients have no first ICU stay at all')
    return stays.astype('int64').to_numpy(), report


TIMING_TIMES = ['icu_intime', 'icu_outtime', 'antibiotic_time', 'culture_time',
                'suspected_infection_time', 'sofa_time', 'deathtime', 'dischtime']


def landmark_table(timing, stays, n_hours=N_HOURS):
    """Times of the Sepsis-3 elements relative to the 24-hour landmark.

    The landmark is 24 hours after the clock hour of ICU admission, the
    end of the last hour that any input can contain. Negative hours are
    before the landmark. A patient is eligible when the antibiotic, the
    culture, and SOFA >= 2 had all occurred by the landmark, so that
    Sepsis-3 could be ascertained at the time of prediction, and the
    patient was alive and still in the ICU then; a missing time makes the
    patient ineligible.
    """
    t = timing[timing['stay_id'].isin(stays)].drop_duplicates('stay_id').copy()
    for c in TIMING_TIMES:
        if c in t:
            t[c] = pd.to_datetime(t[c], errors='coerce')
    t['landmark'] = t['icu_intime'].dt.floor('h') + pd.Timedelta(hours=n_hours)

    def rel(c):
        return (t[c] - t['landmark']).dt.total_seconds() / 3600

    for c, name in [('antibiotic_time', 'antibiotic_h'), ('culture_time', 'culture_h'),
                    ('suspected_infection_time', 'suspicion_h'), ('sofa_time', 'sofa_h')]:
        t[name] = rel(c)
    t['complete_h'] = t[['antibiotic_h', 'culture_h', 'sofa_h']].max(axis=1, skipna=False)
    t['died_by_landmark'] = t['deathtime'].notna() & (t['deathtime'] <= t['landmark'])
    t['left_icu_by_landmark'] = t['icu_outtime'] < t['landmark']
    # At risk at the landmark: alive and still in the ICU.
    at_risk = ~t['died_by_landmark'] & ~t['left_icu_by_landmark']
    t['eligible'] = (t['complete_h'] <= 0) & at_risk
    t['eligible_suspicion_and_sofa'] = (t[['suspicion_h', 'sofa_h']].max(axis=1, skipna=False) <= 0) & at_risk
    t['eligible_sofa_only'] = (t['sofa_h'] <= 0) & at_risk
    t['period'] = pd.to_numeric(t['anchor_year_group'].astype(str)
                                .str.extract(r'(\d{4})')[0], errors='coerce')
    return t


# ======================================================================
# Reconciliation with the cohort table of [15]
# ======================================================================

def reconcile(V, M, X_old, features, hours=None, M_old=None, tol=1e-3):
    """Compare the rebuilt grid with the cohort table of [15], per variable.

    Measured hours (new mask one): whether the table holds the same value,
    and how many are true repeats, i.e. a new measurement equal to the
    previous one. Unmeasured hours after the first measurement: whether
    the table carried the previous value forward or holds a value without
    a record in the rebuilt extraction. Hours before the first measurement: the table
    filled them backward, the rebuild leaves them missing.
    """
    T = V.shape[1]
    sel = np.zeros((V.shape[0], T), dtype=bool)
    sel[:, np.arange(T) if hours is None else np.asarray(hours)] = True
    F = ffill_causal(V)
    rows = []
    for j, f in enumerate(features):
        m = M[:, :, j] == 1
        seen = np.maximum.accumulate(m, axis=1)
        prev_seen = np.zeros_like(seen)
        prev_seen[:, 1:] = seen[:, :-1]
        xo = X_old[:, :, j].astype(float)
        xo_prev = np.full_like(xo, np.nan)
        xo_prev[:, 1:] = xo[:, :-1]
        f_prev = np.full_like(xo, np.nan)
        f_prev[:, 1:] = F[:, :-1, j]
        vn = V[:, :, j].astype(float)

        def same(a, b):
            return np.isclose(a, b, rtol=tol, atol=tol)

        obs = m & sel
        rep = obs & prev_seen
        un = ~m & prev_seen & sel
        before = ~seen & sel
        changed = un & ~same(xo, xo_prev)
        n_all = int(sel.sum())
        rec = {
            'variable': f,
            'hours': n_all,
            'measured hours (%)': 100 * obs.sum() / n_all,
            'measured: table value equal (%)': 100 * np.mean(same(xo[obs], vn[obs])) if obs.any() else np.nan,
            'measured after an earlier one: true repeat (%)': 100 * np.mean(same(vn[rep], f_prev[rep])) if rep.any() else np.nan,
            'unmeasured after first: table carried forward (%)': 100 * (1 - changed.sum() / un.sum()) if un.any() else np.nan,
            'table change without source record (% of hours)': 100 * changed.sum() / n_all,
            'before first measurement (% of hours)': 100 * before.sum() / n_all,
            'never measured in 0-23 h (% of patients)': 100 * np.mean(~seen[:, -1]),
            'rebuilt: unmeasured hours equal to previous (%)':
                100 * np.mean(same(F[:, :, j][un], f_prev[un])) if un.any() else np.nan,
        }
        if M_old is not None:
            rec['old mask equal to new (%)'] = 100 * np.mean((M_old[:, :, j] == 1)[sel] == m[sel])
        rows.append(rec)
    return pd.DataFrame(rows)


# ======================================================================
# Fold-wise preprocessing
# ======================================================================

def fill_values(V, M, rows):
    """Median of the observed hourly values of the training patients,
    the fill for hours before the first observation."""
    J = V.shape[2]
    med = np.zeros(J)
    Vr, Mr = V[rows], M[rows] == 1
    for j in range(J):
        v = Vr[:, :, j][Mr[:, :, j]]
        med[j] = float(np.median(v)) if v.size else 0.0
    return med


def winsor_standardize(X, rows, obs=None, pct=(1, 99), limits='all'):
    """Winsorise at training percentiles, then standardise with the
    training mean and SD, per channel.

    limits='all' estimates the percentiles over every training cell, the
    rule of [15]. limits='nonzero' estimates them over the nonzero cells,
    for channels that are zero by construction wherever nothing changed
    (the change channels), so that a sparse channel is not clipped to a
    constant; zero always stays inside the limits. limits='none' skips
    winsorisation, for channels bounded by construction (mask, time since
    observation, counts). With obs, limits and moments come from observed
    cells only and missing cells stay missing (the GRU-D values). A
    channel with zero training SD is divided by one, so it becomes a
    constant zero."""
    X = np.asarray(X, dtype=np.float32)
    C = X.shape[2]
    tr = X[rows].reshape(-1, C)
    lo, hi, mu, sd, clipped = (np.zeros(C) for _ in range(5))
    if obs is None and limits == 'all':
        lo = np.percentile(tr, pct[0], axis=0)
        hi = np.percentile(tr, pct[1], axis=0)
    elif obs is None and limits == 'nonzero':
        for c in range(C):
            v = tr[tr[:, c] != 0, c]
            if v.size:
                lo[c], hi[c] = np.percentile(v, pct)
        lo, hi = np.minimum(lo, 0.0), np.maximum(hi, 0.0)
    elif obs is None and limits == 'none':
        lo, hi = np.full(C, -np.inf), np.full(C, np.inf)
    if obs is None:
        w = np.clip(tr, lo, hi)
        mu, sd = w.mean(axis=0), w.std(axis=0)
        clipped = ((tr < lo) | (tr > hi)).mean(axis=0)
    else:
        ob = obs[rows].reshape(-1, C) == 1
        for c in range(C):
            v = tr[ob[:, c], c]
            if v.size:
                lo[c], hi[c] = np.percentile(v, pct)
                w = np.clip(v, lo[c], hi[c])
                mu[c], sd[c] = w.mean(), w.std()
                clipped[c] = np.mean((v < lo[c]) | (v > hi[c]))
    const = sd == 0
    sd = np.where(const, 1.0, sd)
    Z = ((np.clip(X, lo, hi) - mu) / sd).astype(np.float32)
    return Z, {'lo': lo, 'hi': hi, 'mu': mu, 'sd': sd, 'constant': const,
               'clipped_share': clipped}


# Channel groups of the CNN-BiGRU inputs: x values, u urine output,
# dx and du carry-forward differences, m mask, dt time since observation,
# d change between observed values, c cumulative number of measurements.
GROUP_SOURCE = {'u': 'U', 'dx': 'DX', 'du': 'DU', 'm': 'M', 'dt': 'DT', 'd': 'D', 'c': 'C'}

REPRESENTATIONS = {
    'R1': ('x', 'u'),
    'R2': ('x', 'u', 'dx', 'du'),
    'R3': ('x', 'u', 'm'),
    'R4': ('x', 'u', 'm', 'dt'),
    'R5': ('x', 'u', 'm', 'dt', 'd'),
    'T': ('m', 'dt'),                  # timing only: no physiological value
    'C': ('x', 'u', 'c'),              # values and cumulative counts
    'TBAL': ('x', 'u', 'm', 'dt'),     # the R4 channels, attention model
}


# Winsorisation limits per channel group. 'paper15' is the rule of [15]
# (every training cell); 'revised' estimates the limits of the change
# channels from their nonzero cells and leaves the bounded timing
# channels unclipped, so that no sparse channel collapses.
LIMITS = {
    'paper15': {g: 'all' for g in ('x', 'u', 'dx', 'du', 'm', 'dt', 'd', 'c')},
    'revised': {'x': 'all', 'u': 'all', 'dx': 'nonzero', 'du': 'nonzero',
                'd': 'nonzero', 'm': 'none', 'dt': 'none', 'c': 'none'},
}


def channel_groups(comp, rows, needed, rule='revised'):
    """Fold-wise groups: x is carried forward, with the training median
    before the first observation; every channel is then winsorised
    (LIMITS[rule]) and standardised on the training patients."""
    out, params = {}, {}
    if 'x' in needed:
        med = fill_values(comp['V'], comp['M'], rows)
        X = np.where(np.isnan(comp['F']), med[None, None, :], comp['F'])
        out['x'], params['x'] = winsor_standardize(X, rows, limits=LIMITS[rule]['x'])
    for g, key in GROUP_SOURCE.items():
        if g in needed:
            out[g], params[g] = winsor_standardize(comp[key], rows, limits=LIMITS[rule][g])
    return out, params


def assemble(groups, names):
    return np.concatenate([groups[g] for g in names], axis=2).astype(np.float32)


def grud_inputs(comp, rows, convention='che', T=N_HOURS):
    """[x | m | delta] for GRU-D and the matched recurrent models.

    x is the last observed value, winsorised and standardised with the
    observed values of the training patients, so that the empirical mean
    of [16] is zero; before the first observation x equals that mean.
    The mask is binary and delta, in hours divided by T, follows the
    convention of [16] ('che') or Eq. (2) ('reset'). Urine output has no
    mask: m = 1 and delta = 0.
    """
    xs, params = winsor_standardize(comp['F'], rows, obs=comp['M'])
    xs = np.nan_to_num(xs, nan=0.0)
    us, _ = winsor_standardize(comp['U'], rows)
    n, Tn, _ = comp['M'].shape
    m = np.concatenate([comp['M'], np.ones((n, Tn, 1), np.float32)], axis=2)
    dt = comp['DTC'] if convention == 'che' else comp['DT']
    dt = np.concatenate([dt / T, np.zeros((n, Tn, 1), np.float32)], axis=2)
    return np.concatenate([xs, us, m, dt], axis=2).astype(np.float32), params


def old_inputs(name, dyn_b, mask_old, rows, n_dyn):
    """Inputs of the first submission: the cohort table of [15], filled
    forward and backward, with the separately reconstructed mask."""
    if name == 'R2_old':
        X = dyn_b
    elif name == 'R3_old':
        X = np.concatenate([dyn_b[:, :, :n_dyn], mask_old], axis=2)
    else:
        raise ValueError(name)
    return winsor_standardize(X, rows)


def static_inputs(num, cat, rows):
    """Numeric static inputs filled with the training median where
    missing, one-hot categories appended, all standardised on the
    training patients as in [15]."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        med = np.nanmedian(num[rows], axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    X = np.hstack([np.where(np.isnan(num), med, num), cat]).astype(np.float64)
    return StandardScaler().fit(X[rows]).transform(X).astype(np.float32)


# ======================================================================
# Models
# ======================================================================

def compile_model(model, lr):
    model.compile(optimizer=keras.optimizers.Adam(lr), loss='binary_crossentropy',
                  metrics=[keras.metrics.AUC(name='auc')])
    return model


def static_pathway(stat_in, hp):
    """Static pathway of [15]: Dense 32, batch normalisation, Dense 16."""
    reg = keras.regularizers.l2(hp['l2'])
    s = layers.Dense(32, activation='relu', kernel_regularizer=reg)(stat_in)
    s = layers.BatchNormalization()(s)
    s = layers.Dropout(hp['dropout'])(s)
    return layers.Dense(16, activation='relu', kernel_regularizer=reg)(s)


def output_head(x, hp):
    """Output head of [15]: Dense 32 and a sigmoid kept in float32."""
    reg = keras.regularizers.l2(hp['l2'])
    x = layers.Dense(32, activation='relu', kernel_regularizer=reg)(x)
    x = layers.Dropout(hp['dropout'])(x)
    return layers.Dense(1, activation='sigmoid', dtype='float32')(x)


def build_static_only(n_static, hp):
    """Static pathway and head alone: the no-values, no-timing cell of
    the factorial comparison."""
    stat_in = layers.Input(shape=(n_static,), name='static')
    model = keras.Model(stat_in, output_head(static_pathway(stat_in, hp), hp))
    return compile_model(model, hp['learning_rate'])


if keras is not None:

    class GRUDCell(keras.layers.Layer):
        """GRU-D cell of Che et al. [16].

        Input per step: [x | m | delta], each n_feat wide, where x holds
        the last observed value (the training mean, zero, before the
        first observation). Input decay is per variable,
            x_hat = m*x + (1-m)*(g_x*x_last + (1-g_x)*x_mean),
        with x_mean = 0, and the hidden state decays as h <- g_h*h, where
        g = exp(-max(0, W*delta + b)). The mask also enters the gates.
        """

        def __init__(self, units, n_feat, l2=0.0, **kwargs):
            super().__init__(**kwargs)
            self.units, self.n_feat, self.l2 = units, n_feat, l2
            self.state_size = units
            self.output_size = units

        def build(self, input_shape):
            D, H = self.n_feat, self.units
            reg = keras.regularizers.l2(self.l2) if self.l2 else None
            # A small positive start keeps the ReLU active so the decay can learn.
            self.w_gx = self.add_weight(shape=(D,), name='w_gx',
                                        initializer=keras.initializers.Constant(0.1))
            self.b_gx = self.add_weight(shape=(D,), name='b_gx', initializer='zeros')
            self.W_gh = self.add_weight(shape=(D, H), name='W_gh',
                                        initializer='glorot_uniform', regularizer=reg)
            self.b_gh = self.add_weight(shape=(H,), name='b_gh', initializer='zeros')
            self.W = self.add_weight(shape=(D, 3 * H), name='W',
                                     initializer='glorot_uniform', regularizer=reg)
            self.V = self.add_weight(shape=(D, 3 * H), name='V',
                                     initializer='glorot_uniform', regularizer=reg)
            self.U = self.add_weight(shape=(H, 3 * H), name='U',
                                     initializer='orthogonal', regularizer=reg)
            self.b = self.add_weight(shape=(3 * H,), name='b', initializer='zeros')
            self.built = True

        def call(self, inputs, states, training=None):
            D, H = self.n_feat, self.units
            h = states[0] if isinstance(states, (list, tuple)) else states
            x, m, d = inputs[:, :D], inputs[:, D:2 * D], inputs[:, 2 * D:]
            g_x = ops.exp(-ops.relu(self.w_gx * d + self.b_gx))
            x_hat = m * x + (1.0 - m) * g_x * x
            g_h = ops.exp(-ops.relu(ops.matmul(d, self.W_gh) + self.b_gh))
            h = g_h * h
            xw = ops.matmul(x_hat, self.W) + ops.matmul(m, self.V) + self.b
            hu = ops.matmul(h, self.U[:, :2 * H])
            z = ops.sigmoid(xw[:, :H] + hu[:, :H])
            r = ops.sigmoid(xw[:, H:2 * H] + hu[:, H:])
            cand = ops.tanh(xw[:, 2 * H:] + ops.matmul(r * h, self.U[:, 2 * H:]))
            h_new = (1.0 - z) * h + z * cand
            return h_new, [h_new]

    class TimeAwareAttention(keras.layers.Layer):
        """Attention over the hours of a window, penalised by the time left
        to the end of the window:
            e_t = v' tanh(W h_t + b) - softplus(lam) * tau_t,
            tau_t = (window - 1 - t) / 24,  alpha = softmax(e).
        Returns sum_t alpha_t h_t. A reading of the time-aware attention
        described for TBAL in [20], whose full specification is in the
        supplementary material of that study."""

        def __init__(self, units, n_hours=N_HOURS, **kwargs):
            super().__init__(**kwargs)
            self.units, self.n_hours = units, n_hours

        def build(self, input_shape):
            H = int(input_shape[-1])
            self.Wa = self.add_weight(shape=(H, self.units), name='Wa', initializer='glorot_uniform')
            self.ba = self.add_weight(shape=(self.units,), name='ba', initializer='zeros')
            self.va = self.add_weight(shape=(self.units, 1), name='va', initializer='glorot_uniform')
            self.lam = self.add_weight(shape=(1,), name='lam', initializer='zeros')
            self.built = True

        def call(self, h):
            steps = h.shape[1]
            tau = ops.cast(ops.flip(ops.arange(steps), axis=0), h.dtype) / float(self.n_hours)
            e = ops.squeeze(ops.matmul(ops.tanh(ops.matmul(h, self.Wa) + self.ba), self.va), axis=-1)
            e = e - ops.softplus(self.lam) * ops.expand_dims(tau, 0)
            a = ops.softmax(e, axis=1)
            return ops.sum(h * ops.expand_dims(a, -1), axis=1)


def _with_static(x, dyn_in, n_static, hp):
    stat_in = layers.Input(shape=(n_static,), name='static')
    x = layers.Concatenate()([x, static_pathway(stat_in, hp)])
    model = keras.Model([dyn_in, stat_in], output_head(x, hp))
    return compile_model(model, hp['learning_rate'])


def build_grud(window, n_in, n_static, hp):
    """One GRU-D layer replacing the convolutional and recurrent pathway."""
    dyn_in = layers.Input(shape=(window, n_in), name='dynamic')
    x = layers.RNN(GRUDCell(int(hp['units']), n_in // 3, hp['l2'], dtype='float32'),
                   dtype='float32', name='grud')(dyn_in)
    x = layers.Dropout(hp['dropout'])(x)
    return _with_static(x, dyn_in, n_static, hp)


def build_rnn(window, n_in, n_static, hp, bidirectional=False):
    """GRU on [x | m | delta] without decay (GRU-Simple of [16]); the
    bidirectional version splits the units between the two directions so
    that the output width matches."""
    reg = keras.regularizers.l2(hp['l2'])
    units = int(hp['units'])
    dyn_in = layers.Input(shape=(window, n_in), name='dynamic')
    if bidirectional:
        x = layers.Bidirectional(layers.GRU(max(units // 2, 1), kernel_regularizer=reg))(dyn_in)
    else:
        x = layers.GRU(units, kernel_regularizer=reg)(dyn_in)
    x = layers.Dropout(hp['dropout'])(x)
    return _with_static(x, dyn_in, n_static, hp)


def build_tbal(window, n_in, n_static, hp):
    """Time-aware bidirectional attention LSTM after [20]: a bidirectional
    LSTM over the hourly values, mask and time since observation, then
    time-aware attention, the static pathway, and the head of [15]."""
    reg = keras.regularizers.l2(hp['l2'])
    dyn_in = layers.Input(shape=(window, n_in), name='dynamic')
    units = int(hp['units'])
    h = layers.Bidirectional(layers.LSTM(units, return_sequences=True,
                                         kernel_regularizer=reg))(dyn_in)
    h = layers.Dropout(hp['dropout'])(h)
    x = TimeAwareAttention(units, dtype='float32', name='time_attention')(h)
    return _with_static(x, dyn_in, n_static, hp)


def build_from_recipe(recipe, window, n_in, n_static, train_module=None):
    """Build a model from a JSON-serialisable recipe {'kind', 'hp'}, so that
    the same model can be built inside a separate training process.
    'cnn_bigru' is the network of [15] (train.build_model)."""
    kind, hp = recipe['kind'], recipe['hp']
    if kind == 'cnn_bigru':
        return train_module.build_model(window, n_in, n_static, hp, 'full')
    if kind == 'static':
        return build_static_only(n_static, hp)
    if kind == 'grud':
        return build_grud(window, n_in, n_static, hp)
    if kind == 'gru':
        return build_rnn(window, n_in, n_static, hp, bidirectional=False)
    if kind == 'bigru':
        return build_rnn(window, n_in, n_static, hp, bidirectional=True)
    if kind == 'tbal':
        return build_tbal(window, n_in, n_static, hp)
    raise ValueError(f'unknown model kind {kind!r}')


# ======================================================================
# Training
# ======================================================================

def logit(p, eps=1e-7):
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


def logistic_fit(lp, y, slope=True, n_iter=100, tol=1e-10):
    """Maximum-likelihood fit of y on a + b*lp, or a + lp with the slope
    fixed at one (calibration-in-the-large)."""
    lp = np.asarray(lp, dtype=float)
    y = np.asarray(y, dtype=float)
    a, b = 0.0, 1.0
    for _ in range(n_iter):
        p = expit(a + b * lp)
        w = p * (1 - p) + 1e-12
        if slope:
            g = np.array([np.sum(y - p), np.sum((y - p) * lp)])
            Hm = np.array([[w.sum(), (w * lp).sum()], [(w * lp).sum(), (w * lp * lp).sum()]])
            step = np.linalg.solve(Hm, g)
            a, b = a + step[0], b + step[1]
            if np.max(np.abs(step)) < tol:
                break
        else:
            step = np.sum(y - p) / w.sum()
            a += step
            if abs(step) < tol:
                break
    return float(a), float(b)


def platt(p_cal, y_cal, p):
    """Platt scaling fitted on the internal partition, Eq. (7)."""
    a, b = logistic_fit(logit(p_cal), y_cal, slope=True)
    return expit(a + b * logit(p)), (a, b)


def class_weights(y):
    pos, n = float(np.sum(y)), float(len(y))
    return {0: n / (2 * (n - pos)), 1: n / (2 * pos)}


def fit_predict(build, dyn, stat, y, fit_rows, val_rows, pred_rows, W, S, seed,
                make_windows, uses_dyn=True, epochs=40, batch_size=512,
                patience=4, verbose=0):
    """Train on the windows of fit_rows, stop early on the right-aligned
    windows of val_rows, and predict the right-aligned window of every
    patient in pred_rows. Returns {name: (rows, probability)}."""
    keras.utils.set_random_seed(seed)
    Xd_tr, Xs_tr, y_tr, _ = make_windows(dyn, stat, y, fit_rows, W, S, training=True)
    Xd_va, Xs_va, y_va, r_va = make_windows(dyn, stat, y, val_rows, W, S, training=False)

    def pack(d, s):
        return [d, s] if uses_dyn else s

    model = build()
    stop = keras.callbacks.EarlyStopping(monitor='val_auc', mode='max', patience=patience,
                                         restore_best_weights=True)
    hist = model.fit(pack(Xd_tr, Xs_tr), y_tr, validation_data=(pack(Xd_va, Xs_va), y_va),
                     epochs=epochs, batch_size=batch_size, class_weight=class_weights(y_tr),
                     callbacks=[stop], verbose=verbose)
    del Xd_tr, Xs_tr
    out = {'val': (r_va, model.predict(pack(Xd_va, Xs_va), batch_size=1024,
                                       verbose=0).ravel().astype(float))}
    for name, rows in pred_rows.items():
        Xd, Xs, _, rr = make_windows(dyn, stat, y, rows, W, S, training=False)
        out[name] = (rr, model.predict(pack(Xd, Xs), batch_size=1024,
                                       verbose=0).ravel().astype(float))
    va = hist.history.get('val_auc', [np.nan])
    info = {'epochs run': len(hist.history['loss']),
            'best epoch': int(np.nanargmax(va)) + 1,
            'parameters': int(model.count_params())}
    keras.backend.clear_session()
    del model
    return out, info


class Context:
    """Everything the runners share: outcome, partitions, windows, files.

    With isolate=True (the default) every model is trained in a separate,
    short-lived Python process. TensorFlow keeps the graph of every traced
    training function for the life of a process (about 70-100 MB per
    model), so hundreds of fits in one session would exhaust the memory of
    a Colab runtime; a process that ends returns all of it. Each fold then
    also starts from a clean state, as in a fresh session.
    """

    def __init__(self, y, pids, W, S, make_windows, pred_dir, log_path, cv, test,
                 temporal=None, epochs=40, batch_size=512, patience=4, device='GPU',
                 src_dir=None, train_module=None, isolate=True, work_dir=None):
        self.y, self.pids, self.W, self.S = y, pids, W, S
        self.make_windows = make_windows
        self.pred_dir, self.log_path = pred_dir, log_path
        self.cv, self.test, self.temporal = cv, test, temporal
        self.epochs, self.batch_size, self.patience = epochs, batch_size, patience
        self.device = device
        self.src_dir, self.train_module, self.isolate = src_dir, train_module, isolate
        self.work_dir = work_dir or os.path.join('/tmp', 'paperA_rev_work')
        os.makedirs(f'{pred_dir}/parts', exist_ok=True)

    def fit(self, spec, dyn, stat, fit_rows, val_rows, pred_rows, seed):
        """Train one model from its recipe, in a separate process or here."""
        uses_dyn = spec['recipe']['kind'] != 'static'
        if not self.isolate:
            return fit_predict(lambda: build_from_recipe(spec['recipe'], self.W, dyn.shape[2],
                                                         stat.shape[1], self.train_module),
                               dyn, stat, self.y, fit_rows, val_rows, pred_rows, self.W, self.S,
                               seed, self.make_windows, uses_dyn, self.epochs, self.batch_size,
                               self.patience)
        return fit_predict_isolated(spec['recipe'], dyn, stat, self.y, fit_rows, val_rows,
                                    pred_rows, self.W, self.S, seed, self.work_dir, self.src_dir,
                                    self.epochs, self.batch_size, self.patience)


def fit_predict_isolated(recipe, dyn, stat, y, fit_rows, val_rows, pred_rows, W, S, seed,
                         work_dir, src_dir, epochs=40, batch_size=512, patience=4):
    """fit_predict in a fresh Python process: the inputs go to a local
    working folder, the process trains and predicts, and the results come
    back through that folder. Same result as fit_predict, without memory
    accumulating across models."""
    os.makedirs(work_dir, exist_ok=True)
    for f in ('out.npz', 'info.json'):
        if os.path.exists(f'{work_dir}/{f}'):
            os.remove(f'{work_dir}/{f}')
    np.save(f'{work_dir}/dyn.npy', np.ascontiguousarray(dyn, dtype=np.float32))
    np.save(f'{work_dir}/stat.npy', np.ascontiguousarray(stat, dtype=np.float32))
    np.save(f'{work_dir}/y.npy', np.asarray(y, dtype=np.float32))
    np.savez(f'{work_dir}/rows.npz', fit=fit_rows, val=val_rows,
             **{f'pred_{k}': v for k, v in pred_rows.items()})
    job = {'recipe': recipe, 'W': int(W), 'S': int(S), 'seed': int(seed), 'epochs': int(epochs),
           'batch_size': int(batch_size), 'patience': int(patience), 'src': src_dir,
           'work': work_dir}
    with open(f'{work_dir}/job.json', 'w') as f:
        json.dump(job, f)
    env = dict(os.environ, TF_CPP_MIN_LOG_LEVEL='2', PYTHONWARNINGS='ignore')
    r = subprocess.run([sys.executable, os.path.abspath(__file__), f'{work_dir}/job.json'],
                       capture_output=True, text=True, env=env)
    if r.returncode != 0 or not os.path.exists(f'{work_dir}/out.npz'):
        raise RuntimeError('training process failed:\n' + (r.stderr or r.stdout)[-4000:])
    z = np.load(f'{work_dir}/out.npz')
    with open(f'{work_dir}/info.json') as f:
        info = json.load(f)
    out = {k[5:]: (z[f'rows_{k[5:]}'], z[k]) for k in z.files if k.startswith('prob_')}
    return out, info


def _run_job(job_path):
    """Entry point of the training process (python paper_a_rev.py job.json)."""
    with open(job_path) as f:
        job = json.load(f)
    sys.path.insert(0, job['src'])
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    for g in gpus:
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass
    # Same precision policy as [15] on GPU; float32 on CPU.
    keras.mixed_precision.set_global_policy('mixed_float16' if gpus else 'float32')
    import windows as wn
    train_module = None
    if job['recipe']['kind'] == 'cnn_bigru':
        import train as train_module
    w = job['work']
    dyn = np.load(f'{w}/dyn.npy', mmap_mode='r')
    stat = np.load(f'{w}/stat.npy')
    y = np.load(f'{w}/y.npy')
    rows = np.load(f'{w}/rows.npz')
    preds = {k[5:]: rows[k] for k in rows.files if k.startswith('pred_')}
    out, info = fit_predict(
        lambda: build_from_recipe(job['recipe'], job['W'], dyn.shape[2], stat.shape[1], train_module),
        dyn, stat, y, rows['fit'], rows['val'], preds, job['W'], job['S'], job['seed'],
        wn.make_windows, job['recipe']['kind'] != 'static', job['epochs'], job['batch_size'],
        job['patience'])
    np.savez(f'{w}/out.npz', **{f'rows_{k}': r for k, (r, _) in out.items()},
             **{f'prob_{k}': p for k, (_, p) in out.items()})
    with open(f'{w}/info.json', 'w') as f:
        json.dump(info, f)


def append_row(path, rec):
    """Append one row to a CSV log, widening the columns when needed."""
    new = pd.DataFrame([rec])
    if os.path.exists(path):
        old = pd.read_csv(path)
        if set(new.columns) - set(old.columns):
            pd.concat([old, new], ignore_index=True).to_csv(path, index=False)
            return
        new = new.reindex(columns=old.columns)
        new.to_csv(path, mode='a', index=False, header=False)
    else:
        new.to_csv(path, index=False)


def run_folds(name, spec, ctx, seed, folds=None, tag='CV'):
    """Cross-validation with early stopping on an internal split of the
    training folds. Each fold is saved when it finishes, so an interrupted
    run resumes at the next fold. With folds=None every fold is run and
    the pooled out-of-fold file is written. Returns the fold AUROCs."""
    keys = sorted(ctx.cv) if folds is None else list(folds)
    aucs = []
    for k in keys:
        part = f'{ctx.pred_dir}/parts/{tag}_{name}_s{seed}_f{k}.npz'
        if not os.path.exists(part):
            fit, inner, ev = ctx.cv[k]
            t0 = time.time()
            dyn, stat = spec['prepare'](fit)
            res, info = ctx.fit(spec, dyn, stat, fit, inner, {'eval': ev}, seed)
            rows, p = res['eval']
            np.savez_compressed(part, rows=rows, y_score=p, fold=np.full(len(rows), k))
            append_row(ctx.log_path, {
                'model': name, 'design': tag, 'seed': seed, 'fold': k,
                'n fit': len(fit), 'n stop': len(inner), 'n eval': len(ev),
                'auroc': roc_auc_score(ctx.y[rows], p),
                'auprc': average_precision_score(ctx.y[rows], p), **info,
                'elapsed s': round(time.time() - t0, 1), 'device': ctx.device})
            del dyn, stat
        z = np.load(part)
        aucs.append(roc_auc_score(ctx.y[z['rows']], z['y_score']))
    if folds is None:
        out = f'{ctx.pred_dir}/{tag}_{name}_s{seed}.npz'
        if not os.path.exists(out):
            zs = [np.load(f'{ctx.pred_dir}/parts/{tag}_{name}_s{seed}_f{k}.npz') for k in keys]
            rows = np.concatenate([z['rows'] for z in zs])
            np.savez_compressed(out, subject_id=ctx.pids[rows], y_true=ctx.y[rows],
                                y_score=np.concatenate([z['y_score'] for z in zs]),
                                fold=np.concatenate([z['fold'] for z in zs]))
    return aucs


def run_holdout(name, spec, ctx, seed, design='TEST'):
    """Train on 85% of the development patients, stop early and fit Platt
    scaling on the other 15%, and predict the held-out patients, who play
    no part in either step. design is 'TEST' (random held-out set) or
    'TEMP' (later calendar period)."""
    f_unc = f'{ctx.pred_dir}/{design}_{name}_uncal_s{seed}.npz'
    f_pl = f'{ctx.pred_dir}/{design}_{name}_platt_s{seed}.npz'
    if os.path.exists(f_unc) and os.path.exists(f_pl):
        return
    fit, cal, ev = ctx.test if design == 'TEST' else ctx.temporal
    t0 = time.time()
    dyn, stat = spec['prepare'](fit)
    res, info = ctx.fit(spec, dyn, stat, fit, cal, {'eval': ev}, seed)
    r_cal, p_cal = res['val']
    rows, p = res['eval']
    p_pl, (a, b) = platt(p_cal, ctx.y[r_cal], p)
    for path, score in ((f_unc, p), (f_pl, p_pl)):
        np.savez_compressed(path, subject_id=ctx.pids[rows], y_true=ctx.y[rows], y_score=score)
    append_row(ctx.log_path, {
        'model': name, 'design': design, 'seed': seed, 'fold': 0,
        'n fit': len(fit), 'n stop': len(cal), 'n eval': len(ev),
        'auroc': roc_auc_score(ctx.y[rows], p_pl),
        'auprc': average_precision_score(ctx.y[rows], p_pl),
        'platt a': a, 'platt b': b, **info,
        'elapsed s': round(time.time() - t0, 1), 'device': ctx.device})


def sample_grid(space, n, seed):
    """n distinct configurations drawn from a discrete grid with a fixed
    seed, the search budget of [15] (fifteen draws)."""
    keys = sorted(space)
    grid = list(itertools.product(*(space[k] for k in keys)))
    pick = np.random.default_rng(seed).choice(len(grid), size=min(n, len(grid)), replace=False)
    return [dict(zip(keys, grid[i])) for i in pick]


def run_search(name, make_spec, ctx, space, out_csv, n_trials=15, folds=(1, 2, 3),
               seed=42, search_seed=2026):
    """Random search with the budget of [15]: n_trials draws, one seed,
    three development folds, mean AUROC as the objective. Resumes."""
    results = []
    for i, hp in enumerate(sample_grid(space, n_trials, search_seed)):
        aucs = run_folds(name, make_spec(hp), ctx, seed, folds=folds, tag=f'SEARCH{i:02d}')
        results.append({'trial': i, **hp, 'mean AUROC': float(np.mean(aucs)),
                        **{f'fold {k}': a for k, a in zip(folds, aucs)}})
        pd.DataFrame(results).to_csv(out_csv, index=False)
    res = pd.DataFrame(results)
    best = res.sort_values(['mean AUROC', 'trial'], ascending=[False, True]).iloc[0]
    # A row of mixed columns comes back as floats; restore each grid type.
    return res, {k: type(space[k][0])(best[k]) for k in space}


# ======================================================================
# Statistics
# ======================================================================

def pooled(pred_dir, pattern, seeds):
    """Average the probabilities across seeds, one score per patient, the
    single aggregation convention of [15]."""
    scores, y = [], None
    for s in seeds:
        z = np.load(f"{pred_dir}/{pattern.format(seed=s)}")
        scores.append(pd.Series(z['y_score'], index=z['subject_id']))
        y = pd.Series(z['y_true'], index=z['subject_id'])
    S = pd.concat(scores, axis=1).dropna()
    return pd.DataFrame({'y': y.loc[S.index].astype(int).to_numpy(),
                         'score': S.mean(axis=1).to_numpy()}, index=S.index).sort_index()


def available(pred_dir, pattern, seeds):
    return all(os.path.exists(f"{pred_dir}/{pattern.format(seed=s)}") for s in seeds)


def delong(y, S):
    """AUROCs of several scores on the same patients and the DeLong
    covariance of the estimates."""
    y = np.asarray(y).astype(bool)
    S = np.atleast_2d(np.asarray(S, dtype=float))
    X, Y = S[:, y], S[:, ~y]
    m, n = X.shape[1], Y.shape[1]
    tx = np.vstack([rankdata(r) for r in X])
    ty = np.vstack([rankdata(r) for r in Y])
    tz = np.vstack([rankdata(np.r_[a, b]) for a, b in zip(X, Y)])
    auc = (tz[:, :m].sum(axis=1) / m - (m + 1) / 2) / n
    v10 = (tz[:, :m] - tx) / n
    v01 = 1 - (tz[:, m:] - ty) / m
    cov = np.atleast_2d(np.cov(v10)) / m + np.atleast_2d(np.cov(v01)) / n
    return auc, cov


def auc_contrast(y, scores, weights, margin=0.01):
    """A linear contrast of AUROCs (for example a - b, or an interaction)
    with its DeLong SE, 95% and 90% intervals, the two-sided p, and the
    two one-sided tests at +/- margin."""
    names = list(weights)
    L = np.array([weights[k] for k in names], dtype=float)
    auc, cov = delong(y, np.vstack([scores[k] for k in names]))
    est = float(L @ auc)
    se = float(np.sqrt(max(float(L @ cov @ L), 0.0)))
    if se == 0:
        return {'estimate': est, 'se': 0.0, 'p': 1.0, 'tost p': np.nan,
                '95% low': est, '95% high': est, '90% low': est, '90% high': est}
    return {'estimate': est, 'se': se, 'p': float(2 * norm.sf(abs(est) / se)),
            '95% low': est - 1.96 * se, '95% high': est + 1.96 * se,
            '90% low': est - 1.645 * se, '90% high': est + 1.645 * se,
            'tost p': float(max(norm.sf((est + margin) / se), norm.cdf((est - margin) / se)))}


class _AP:
    """Average precision for bootstrap weights without re-sorting: the
    estimator of sklearn's average_precision_score (step-wise, no
    interpolation), so point estimates and resamples use one definition."""

    def __init__(self, y, s):
        order = np.argsort(-np.asarray(s, dtype=float), kind='mergesort')
        s_sorted = np.asarray(s, dtype=float)[order]
        self.order, self.y = order, np.asarray(y, dtype=float)[order]
        self.ends = np.flatnonzero(np.r_[np.diff(s_sorted) != 0, True])

    def __call__(self, w):
        ws = w[self.order]
        yw = ws * self.y
        tp = np.cumsum(yw)[self.ends]
        fp = np.cumsum(ws - yw)[self.ends]
        if tp[-1] <= 0:
            return np.nan
        prec = tp / np.maximum(tp + fp, 1e-12)
        rec = tp / tp[-1]
        return float(np.sum(np.diff(np.r_[0.0, rec]) * prec))


def ap_contrast(y, scores, weights, B=2000, seed=42):
    """A linear contrast of AUPRCs with a paired percentile bootstrap
    interval (the same resampled patients for every model)."""
    names = list(weights)
    L = np.array([weights[k] for k in names], dtype=float)
    y = np.asarray(y).astype(int)
    aps = [_AP(y, scores[k]) for k in names]
    est = float(L @ np.array([average_precision_score(y, scores[k]) for k in names]))
    rng = np.random.default_rng(seed)
    n, diffs = len(y), []
    for _ in range(B):
        i = rng.integers(0, n, n)
        if 0 < y[i].sum() < n:
            w = np.bincount(i, minlength=n).astype(float)
            diffs.append(float(L @ np.array([f(w) for f in aps])))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return est, float(lo), float(hi)


def benjamini_hochberg(p_values):
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adjusted = np.empty(n)
    running = 1.0
    for rank in range(n - 1, -1, -1):
        running = min(running, p[order[rank]] * n / (rank + 1))
        adjusted[order[rank]] = running
    return adjusted


def calibration(y, p):
    """Calibration slope and intercept (logistic recalibration) and
    calibration-in-the-large (intercept with the slope fixed at one)."""
    lp = logit(p)
    a, b = logistic_fit(lp, y, slope=True)
    citl, _ = logistic_fit(lp, y, slope=False)
    return {'calibration slope': b, 'calibration intercept': a, 'calibration-in-the-large': citl}


def performance(y, p, B=2000, seed=2026, ci=True):
    """Discrimination, overall accuracy, and calibration of one score;
    with ci, a DeLong interval for the AUROC and bootstrap intervals for
    the AUPRC, calibration slope, and calibration-in-the-large."""
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    auc, cov = delong(y, p)
    rec = {'n': len(y), 'deaths': int(y.sum()), 'AUROC': float(auc[0]),
           'AUPRC': float(average_precision_score(y, p)),
           'Brier': float(brier_score_loss(y, p)), **calibration(y, p)}
    if ci:
        se = float(np.sqrt(cov[0, 0]))
        rec['AUROC 95% CI'] = f"{auc[0] - 1.96 * se:.4f} to {auc[0] + 1.96 * se:.4f}"
        rng = np.random.default_rng(seed)
        ap = _AP(y, p)
        lp = logit(p)
        boot = []
        for _ in range(B):
            i = rng.integers(0, len(y), len(y))
            if 0 < y[i].sum() < len(y):
                w = np.bincount(i, minlength=len(y)).astype(float)
                a_, b_ = logistic_fit(lp[i], y[i], slope=True)
                c_, _ = logistic_fit(lp[i], y[i], slope=False)
                boot.append((ap(w), b_, c_))
        boot = np.array(boot)
        for j, key in enumerate(('AUPRC', 'calibration slope', 'calibration-in-the-large')):
            lo, hi = np.percentile(boot[:, j], [2.5, 97.5])
            rec[f'{key} 95% CI'] = f"{lo:.3f} to {hi:.3f}"
    return rec


# Operating points: share of patients flagged (alert budget) or a risk
# threshold. Net benefit weights false positives by the odds of the risk
# threshold p_t, so it is defined for the risk thresholds only; a budget
# cutoff is reported with sensitivity and PPV.
POINTS = [('top 10%', 'budget', 0.10), ('top 20%', 'budget', 0.20),
          ('risk >= 0.15', 'risk', 0.15), ('risk >= 0.30', 'risk', 0.30),
          ('risk >= 0.50', 'risk', 0.50)]


def _flags(p, kind, val):
    if kind == 'risk':
        return p >= val
    return p >= np.quantile(p, 1 - val, axis=-1, keepdims=True)


def _tp_fp(f, yy):
    return (f & (yy == 1)).sum(-1), (f & (yy == 0)).sum(-1)


def net_benefit(tp, fp, n, pt):
    """Net benefit of decision curve analysis per 1000 patients at the
    risk threshold pt."""
    return 1000 * (tp / n - fp / n * pt / (1 - pt))


def operating_points(y, P, ref='R2', points=POINTS, B=2000, seed=2026):
    """Each model at each operating point, and its paired difference from
    the reference with percentile bootstrap intervals."""
    y = np.asarray(y).astype(np.int8)
    n, n_pos = len(y), int(y.sum())
    rows = []
    for m, p in P.items():
        for name, kind, val in points:
            f = _flags(p, kind, val)
            tp, fp = _tp_fp(f, y)
            rec = {'model': m, 'operating point': name, 'flagged (%)': 100 * f.mean(),
                   'sensitivity (%)': 100 * tp / n_pos,
                   'specificity (%)': 100 * (n - n_pos - fp) / (n - n_pos),
                   'PPV (%)': 100 * tp / max(tp + fp, 1)}
            if kind == 'risk':
                rec['net benefit per 1000'] = net_benefit(tp, fp, n, val)
            rows.append(rec)
    each = pd.DataFrame(rows)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(B, n))
    yb = y[idx]
    yb_pos = np.maximum(yb.sum(1), 1)

    def boot(p, kind, val):
        f = _flags(p[idx], kind, val)
        tp, fp = _tp_fp(f, yb)
        nb = net_benefit(tp, fp, n, val) if kind == 'risk' else None
        return 100 * tp / yb_pos, 100 * tp / np.maximum(tp + fp, 1), nb

    def ci(a):
        return f'{np.percentile(a, 2.5):+.1f} to {np.percentile(a, 97.5):+.1f}'

    rows = []
    for name, kind, val in points:
        f_ref = _flags(P[ref], kind, val)
        tp_r, fp_r = _tp_fp(f_ref, y)
        s_r, v_r, nb_r = boot(P[ref], kind, val)
        for m, p in P.items():
            if m == ref:
                continue
            f = _flags(p, kind, val)
            tp, fp = _tp_fp(f, y)
            s_m, v_m, nb_m = boot(p, kind, val)
            rec = {'model': m, 'operating point': name,
                   f'flags differing from {ref} (%)': 100 * np.mean(f != f_ref),
                   'difference in deaths flagged per 1000': 1000 * (tp - tp_r) / n,
                   'difference in survivors flagged per 1000': 1000 * (fp - fp_r) / n,
                   'difference in sensitivity (pp)': 100 * (tp - tp_r) / n_pos,
                   'sensitivity 95% CI': ci(s_m - s_r),
                   'difference in PPV (pp)': 100 * (tp / max(tp + fp, 1) - tp_r / max(tp_r + fp_r, 1)),
                   'PPV 95% CI': ci(v_m - v_r)}
            if kind == 'risk':
                rec['difference in net benefit per 1000'] = net_benefit(tp, fp, n, val) - net_benefit(tp_r, fp_r, n, val)
                rec['net benefit 95% CI'] = ci(nb_m - nb_r)
            rows.append(rec)
    return each, pd.DataFrame(rows)


def net_benefit_curves(y, P, thresholds=None):
    """Net benefit per 1000 patients across risk thresholds."""
    thresholds = np.round(np.arange(0.05, 0.501, 0.01), 2) if thresholds is None else thresholds
    y = np.asarray(y).astype(np.int8)
    out = {}
    for m, p in P.items():
        vals = []
        for t in thresholds:
            tp, fp = _tp_fp(p >= t, y)
            vals.append(net_benefit(tp, fp, len(y), t))
        out[m] = vals
    return pd.DataFrame(out, index=pd.Index(thresholds, name='threshold'))


def sha256(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


# Query that produces sepsis3_timing.csv (Cell 4), kept here for the release.
SEPSIS3_TIMING_SQL = """-- Sepsis-3 timing and calendar period for every Sepsis-3 ICU stay.
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
"""


if __name__ == '__main__' and len(sys.argv) > 1 and sys.argv[1].endswith('.json'):
    _run_job(sys.argv[1])
