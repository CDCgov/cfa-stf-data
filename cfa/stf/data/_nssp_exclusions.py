"""Read published NSSP reviews and flag observations without removing them."""

import datetime as dt
import warnings

import polars as pl
from cfa.dataops import datacat

from ._utils import _version_spec, _version_to_datetime

_SCHEMA = {
    "reference_date": pl.Date,
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
        ``get_nssp(exclusion_as_of=...)``. If None, resolves the most recent
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
    reference_date: dt.date, exclusion_as_of: dt.date | None
) -> pl.DataFrame | None:
    endpoint = datacat.public.stf.nssp_exclusions.load
    version = endpoint.resolve_version(
        version_spec=_version_spec(exclusion_as_of)
    ).version
    if version is None:
        warnings.warn(
            f"No published NSSP exclusion snapshot matches exclusion_as_of={exclusion_as_of!r}; "
            "returning exclude=False.",
            stacklevel=3,
        )
        return None
    frame = endpoint.get_dataframe(output="polars", version_spec=version)
    if dict(frame.schema) != _SCHEMA:
        raise ValueError(f"NSSP exclusion snapshot must have schema {_SCHEMA}")

    review = frame.filter(pl.col("reference_date") == reference_date)
    if review.is_empty():
        warnings.warn(
            f"NSSP exclusion snapshot {version!r} has no review for "
            f"reference date {reference_date}; returning exclude=False.",
            stacklevel=3,
        )
    else:
        if any(review.null_count().row(0)):
            raise ValueError(f"NSSP review {reference_date} contains null values")
        if review.get_column("state_abb").is_duplicated().any():
            raise ValueError(f"NSSP review {reference_date} contains duplicate scopes")
        dates = pl.col("excluded_dates")
        if review.select(
            dates.list.eval(pl.element().is_null()).list.any().any()
        ).item():
            raise ValueError(f"NSSP review {reference_date} contains null dates")
        if review.select(
            (
                (dates.list.n_unique() != dates.list.len())
                | (dates != dates.list.sort())
            ).any()
        ).item():
            raise ValueError(
                f"NSSP review {reference_date} must contain sorted, unique dates"
            )

    return (
        review.select("state_abb", "excluded_dates")
        .explode("excluded_dates", empty_as_null=False)
        .rename({"excluded_dates": "date"})
        .with_columns(pl.lit(True).alias("exclude"))
    )
