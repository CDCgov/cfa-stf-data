"""Read published NSSP reviews and flag observations without removing them."""

import datetime as dt
import warnings

import polars as pl
from cfa.dataops import datacat

from ._utils import _version_spec, _version_to_datetime

_SCHEMA = {
    "report_date": pl.Date,
    "state_abb": pl.String,
    "excluded_dates": pl.List(pl.Date),
}


def resolve_nssp_exclusion_version(
    as_of: dt.date | None = None,
) -> dt.datetime | str | None:
    """Resolve the data-review snapshot that ``get_nssp`` would load.

    Parameters
    ----------
    as_of
        The latest catalog version date to consider, corresponding to
        ``get_nssp(exclusions_as_of=...)``. If None, resolves the most recent
        available version.

    Returns
    -------
    datetime.datetime | str | None
        The selected catalog version converted to a datetime when possible,
        or None if no version matches.
    """
    version = datacat.public.stf.nssp_exclusions.load.resolve_version(
        version_spec=_version_spec(as_of)
    ).version
    return _version_to_datetime(version)


def _review_flags(
    report_date: dt.date, exclusions_as_of: dt.date | None
) -> pl.DataFrame | None:
    endpoint = datacat.public.stf.nssp_exclusions.load
    version = endpoint.resolve_version(
        version_spec=_version_spec(exclusions_as_of)
    ).version
    if version is None:
        warnings.warn(
            f"No published NSSP exclusion snapshot matches exclusions_as_of={exclusions_as_of!r}; "
            "returning exclude=False.",
        )
        return None
    frame = endpoint.get_dataframe(output="polars", version_spec=version)
    if dict(frame.schema) != _SCHEMA:
        raise ValueError(f"NSSP exclusion snapshot must have schema {_SCHEMA}")

    review = frame.filter(pl.col("report_date") == report_date)
    if review.is_empty():
        warnings.warn(
            f"NSSP exclusion snapshot {version!r} has no review for "
            f"report date {report_date}; returning exclude=False.",
        )

    return (
        review.select("state_abb", "excluded_dates")
        .explode("excluded_dates", empty_as_null=False)
        .rename({"excluded_dates": "date"})
        .with_columns(pl.lit(True).alias("exclude"))
    )
