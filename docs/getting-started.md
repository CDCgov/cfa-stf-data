# Getting started

## Install

The package requires Python 3.12 or later.
Install the repository environment with [uv](https://docs.astral.sh/uv/):

```shell
uv sync
```

All public helpers are importable from `cfa.stf.data`.

## Retrieve NHSN hospital admissions

```python
import datetime as dt

from cfa.stf.data import get_nhsn_hrd

admissions = get_nhsn_hrd(
    disease="flu",
    state_abb=["CA", "OR", "WA"],
    start_date=dt.date(2025, 10, 1),
    end_date=dt.date(2026, 3, 31),
    lazy=False,
)
```

The eager result contains `date`, `state_abb`, `disease`, `target_type`, and `value`.
For NHSN data, `target_type` is `"wk inc hosp"`.

## Retrieve NSSP emergency department visits

```python
import polars as pl

from cfa.stf.data import get_nssp

visits = get_nssp(
    disease=["covid", "flu", "rsv"],
    state_abb="US",
)

visits = visits.filter(
    # Add downstream Polars expressions before evaluating the query.
    pl.col("value") >= 0
).collect()
```

`get_nssp` returns a `polars.LazyFrame` unless `lazy=False`.
Its columns are `date`, `state_abb`, `disease`, `target_type`, `value`, and `exclude`.
`exclude` is always Boolean and defaults to `False` when exclusions are disabled or missing.
For NSSP data, `target_type` is `"inc ed visits"`.

!!! note "National NSSP values"
    When `state_abb="US"`, values are computed by aggregating the available geographic rows in the selected catalog dataset and vintage.

## Apply published data-review exclusions

```python
from cfa.stf.data import get_nssp

visits = get_nssp(
    disease="flu",
    state_abb="CA",
    get_exclusions=True,
)
```

This populates the Boolean `exclude` column without removing observations.
Exclusions apply across diseases for each location and observation date.
US flags use the US review; state exclusions do not change national counts.
The default return type is still lazy; pass `lazy=False` for a DataFrame.

The latest `report_date` in the loaded NSSP data selects the corresponding review in the snapshot, before applying observation filters.
Exclusions snapshots contain `report_date: Date`, `state_abb: String`, and `excluded_dates: List[Date]`.
Use `as_of` to select a historical NSSP vintage:

```python
import datetime as dt

from cfa.stf.data import resolve_nssp_exclusion_version

exclusions_as_of = dt.date(2026, 10, 2)
selected_version = resolve_nssp_exclusion_version(as_of=exclusions_as_of)
visits = get_nssp(
    as_of=dt.date(2026, 10, 1),
    get_exclusions=True,
    exclusions_as_of=exclusions_as_of,
    lazy=False,
)
```

If the selected NSSP vintage is older than `as_of`, its latest `report_date` still determines the review date.
`exclusions_as_of` independently selects the newest published exclusions snapshot at or before its cutoff, using the same convention as the library's other `as_of` arguments.
A date denotes midnight at the start of that day; a datetime preserves the time of day, and the cutoff is inclusive.
Omitting `exclusions_as_of` selects the latest snapshot, even when `as_of` selects a historical NSSP vintage and review date.
Use `resolve_nssp_exclusion_version(as_of=exclusions_as_of)` to record the selected snapshot version.
Like the other version resolvers, it returns a datetime when possible, otherwise the version string, or `None` if no version matches.
If no snapshot matches the cutoff, the function warns and returns `exclude=False` for every observation.
Record the NSSP vintage separately for reproducibility.

If the snapshot lacks the requested review date, the function warns and returns `exclude=False` for every observation.
Missing location reviews and observation dates absent from the exclusion lists receive `exclude=False`.
Empty exclusion lists also produce false flags.
Review rows for locations absent from the result are allowed.
The snapshot's column names and types are checked before selecting the requested review date.

Callers decide whether to filter flagged rows.
See the [NSSP API reference](api/nssp.md).

## Preview the documentation

```shell
uv run --group docs zensical serve
```

Build the static site with:

```shell
uv run --group docs zensical build --clean
```
