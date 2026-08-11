---
hide:
  - navigation
---

# CFA STF Data

Data access helpers for the Center for Forecasting and Outbreak Analytics
Short-Term Forecasting Team.

The `cfa.stf.data` package provides a consistent Polars interface to public
forecasting inputs in the CFA data catalog:

- NHSN weekly hospital respiratory admissions
- NSSP daily emergency department visits
- generation interval, delay, and right-truncation probability mass functions
- NSSP tail-exclusion rules for incomplete or anomalous recent observations

[Get started](getting-started.md){ .md-button .md-button--primary }
[API reference](api/nhsn.md){ .md-button }

## Design

The helpers expose explicit temporal controls. Observation dates select rows;
`as_of` selects the catalog vintage available at a given date. Data-frame
functions return a `polars.LazyFrame` by default so callers can extend and
optimize the query before collecting it.

```python
import datetime as dt

from cfa.stf.data import get_nssp

nssp = get_nssp(
    disease=["Influenza", "RSV"],
    loc_abb="CA",
    as_of=dt.date(2026, 1, 15),
    start_date=dt.date(2025, 10, 1),
)

data = nssp.collect()
```

## Package areas

| Area | Primary functions | Output |
| --- | --- | --- |
| NHSN | [`get_nhsn_hrd`][cfa.stf.data.get_nhsn_hrd] | Weekly admissions in long format |
| NSSP | [`get_nssp`][cfa.stf.data.get_nssp] | Daily emergency department visit counts |
| Tail exclusion | [`get_nssp_with_exclusion`][cfa.stf.data.get_nssp_with_exclusion.get_nssp_with_exclusion] | NSSP data with an `exclude` indicator |
| Nowcast parameters | [`get_nnh_generation_interval_pmf`][cfa.stf.data.get_nnh_generation_interval_pmf] and related functions | Probability mass functions as `list[float]` |
