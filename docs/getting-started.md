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
from cfa.stf.data import get_nssp

visits = get_nssp(
    disease="flu",
    state_abb="CA",
    apply_data_review_exclusions=True,
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

from cfa.stf.data import resolve_nssp_exclusion_version

exclusion_as_of = dt.date(2026, 10, 2)
selected_version = resolve_nssp_exclusion_version(as_of=exclusion_as_of)
visits = get_nssp(
    as_of=dt.date(2026, 10, 1),
    apply_data_review_exclusions=True,
    exclusion_as_of=exclusion_as_of,
    lazy=False,
)
```

An explicit `as_of` selects that review date even if the available NSSP vintage is older.
`exclusion_as_of` independently selects the newest published exclusions snapshot at or before its cutoff, using the same convention as the library's other `as_of` arguments.
A date denotes midnight at the start of that day; a datetime preserves the time of day, and the cutoff is inclusive.
Omitting `exclusion_as_of` selects the latest snapshot, even when `as_of` selects a historical NSSP vintage and review date.
Use `resolve_nssp_exclusion_version(as_of=exclusion_as_of)` to record the selected snapshot version.
Like the other version resolvers, it returns a datetime when possible, otherwise the version string, or `None` if no version matches.
Loading exclusions raises an error if no snapshot matches the cutoff.
Record the NSSP vintage separately for reproducibility.

If the snapshot lacks the requested review date, the function warns and returns `exclude=False` for every observation.
A represented review must contain a row for every location in the filtered NSSP result.
An empty exclusion list counts as reviewed; a missing location row raises an error.
Review rows for locations absent from the result are allowed.
Malformed reviews for the requested date raise an error; other review dates are not validated.

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
