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

The eager result contains `date`, `state_abb`, `disease`, and `hospital_admissions`.

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
Its columns are `date`, `disease`, `state_abb`, and `value`.

!!! note "National NSSP values"
    When `state_abb="US"`, values are computed by aggregating the available geographic rows in the selected catalog dataset and vintage.

## Flag an incomplete tail

```python
from cfa.stf.data import get_nssp_with_exclusion

visits = get_nssp_with_exclusion(
    disease="flu",
    state_abb="CA",
    exclusion_strategy="tail_by_n",
    n=3,
)
```

This eager result adds a Boolean `exclude` column.
Automatic strategies can instead detect a discontinuity in the recent tail using the target disease, the sum of the three respiratory diseases, or the NSSP total series.
See the [NSSP API reference](api/nssp.md) for all strategy arguments.

## Preview the documentation

```shell
uv run --group docs zensical serve
```

Build the static site with:

```shell
uv run --group docs zensical build --clean
```
