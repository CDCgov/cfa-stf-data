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
Its columns are `date`, `state_abb`, `disease`, `target_type`, and `value`.
For NSSP data, `target_type` is `"inc ed visits"`.

!!! note "National NSSP values"
    When `state_abb="US"`, values are computed by aggregating the available geographic rows in the selected catalog dataset and vintage.

## Apply published data-review exclusions

```python
from cfa.stf.data import get_nssp, resolve_nssp_exclusion_version

exclusion_version = resolve_nssp_exclusion_version()
visits = get_nssp(
    disease="flu",
    state_abb="CA",
    apply_data_review_exclusions=True,
    exclusion_version=exclusion_version,
)
```

The result adds a Boolean `exclude` column without removing observations.
Exclusions apply across diseases for each location and observation date.
US flags use the US review; state exclusions do not change national counts.
The default return type is still lazy; pass `lazy=False` for a DataFrame.

Without `as_of`, the latest NSSP vintage is resolved and its calendar date selects the corresponding review in the snapshot.
To select a particular review date:

```python
import datetime as dt

visits = get_nssp(
    as_of=dt.date(2026, 10, 1),
    apply_data_review_exclusions=True,
    exclusion_version=exclusion_version,
    lazy=False,
)
```

An explicit `as_of` selects that review date even if the available NSSP vintage is older.
Omitting `exclusion_version` uses the newest published exclusions snapshot, independently of `as_of`.
Record the exact string returned by `resolve_nssp_exclusion_version()` and supply it on subsequent calls to reload the same exclusions snapshot.
Record the NSSP vintage separately for reproducibility.

If the snapshot lacks the requested review date, the function warns and returns `exclude=False` for every observation.
A represented review must cover all 50 states, DC, and US, even for a query requesting a single location.
Malformed reviews for the requested date raise an error; other review dates are not validated.
Locations outside that supported set receive `exclude=False`.

`get_nssp_with_exclusion()` and its manual and computed tail strategies have been removed.
Use the opt-in flags above; callers decide whether to filter flagged rows.
See the [NSSP API reference](api/nssp.md).

## Preview the documentation

```shell
uv run --group docs zensical serve
```

Build the static site with:

```shell
uv run --group docs zensical build --clean
```
