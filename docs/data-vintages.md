# Data vintages

The observation date and the catalog vintage represent different time axes.
Keeping them distinct is important for retrospective forecast evaluation.

## Observation interval

`start_date` and `end_date` filter the dates represented by rows in the output.
Both bounds are inclusive.

## Information cutoff

For NHSN and NSSP data, `as_of=d` selects the latest catalog version whose
version timestamp is no later than date \(d\). If `as_of=None`, the latest
available version is selected.

```python
import datetime as dt

from cfa.stf.data import get_nssp, resolve_nssp_version

cutoff = dt.date(2025, 12, 1)
version = resolve_nssp_version(as_of=cutoff)
data = get_nssp(as_of=cutoff)
```

The `resolve_*_version` functions make the selected provenance explicit without
loading the corresponding data frame.

## Nowcast parameter validity

Nowcast parameter helpers interpret `as_of` as a validity date and retain rows
satisfying

\[
  \texttt{start\_date} \leq \texttt{as\_of} < \texttt{end\_date}.
\]

Right-truncation parameters have a second temporal coordinate,
`reference_date`. Among parameter estimates available for the selected
location and validity interval, the helper selects the greatest parameter
reference date not exceeding the requested `reference_date`.

