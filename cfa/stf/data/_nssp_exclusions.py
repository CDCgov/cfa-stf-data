"""Read published NSSP reviews and flag observations without removing them."""

import datetime as dt
import warnings

import polars as pl
from cfa.dataops import datacat

_SCHEMA = {
    "reference_date": pl.Date,
    "state_abb": pl.String,
    "excluded_dates": pl.List(pl.Date),
}


def resolve_nssp_exclusion_version(exclusion_version: str | None = None) -> str:
    """Resolve an exact published NSSP data-review snapshot version.

    Parameters
    ----------
    exclusion_version
        An opaque exact catalog version, or None to select the newest published
        snapshot. Published snapshot identifiers sort chronologically.

    Returns
    -------
    str
        The unchanged version identifier, suitable for recording and passing to
        ``get_nssp(exclusion_version=..., apply_data_review_exclusions=True)``.

    Raises
    ------
    ValueError
        If no snapshot exists or the supplied exact version is unavailable.
    """
    versions = datacat.public.stf.nssp_exclusions.load.get_versions()
    if exclusion_version is not None:
        if exclusion_version not in versions:
            raise ValueError(f"NSSP exclusion version not found: {exclusion_version!r}")
        return exclusion_version
    if not versions:
        raise ValueError("No published NSSP exclusion snapshots are available")
    return max(versions)


def _review_flags(
    reference_date: dt.date, exclusion_version: str | None
) -> pl.DataFrame:
    version = resolve_nssp_exclusion_version(exclusion_version)
    endpoint = datacat.public.stf.nssp_exclusions.load
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
        # Keep reviewed locations with empty lists for coverage validation at
        # the NSSP join. Only an absent reference date produces an empty frame.
        .explode("excluded_dates", empty_as_null=True)
        .rename({"excluded_dates": "date"})
        .with_columns(pl.lit(True).alias("exclude"))
    )
