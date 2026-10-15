import datetime as dt
import warnings
from collections.abc import Iterable
from typing import Literal, overload

import polars as pl
from cfa.dataops import datacat

from ._nssp_exclusions import _review_flags
from ._utils import (
    _version_spec,
    _version_to_datetime,
    canonical_disease_expr,
    canonicalize_diseases,
    ensure_list,
)

NSSPDataset = Literal["gold", "comprehensive"]


def _get_nssp_dataset(dataset: NSSPDataset):
    dataset_map = {
        "gold": datacat.public.stf.nssp_gold_v1,
        "comprehensive": datacat.public.stf.comprehensive_nssp_gold,
    }

    if not (datacat_dataset := dataset_map.get(dataset)):
        raise ValueError(
            f"Invalid dataset: {dataset!r}. Expected one of: {set(dataset_map)!r}."
        )
    return datacat_dataset


def resolve_nssp_version(
    dataset: NSSPDataset = "gold",
    as_of: dt.date | None = None,
) -> dt.datetime | str | None:
    """Resolve the catalog version that [`get_nssp`][cfa.stf.data.get_nssp]
    would load.

    Parameters
    ----------
    dataset
        One of the two NSSP datasets: ``"gold"`` or ``"comprehensive"``.
    as_of
        The latest catalog version date to consider. If None, resolves the
        most recent available version.

    Returns
    -------
    datetime.datetime | str | None
        The selected catalog version converted to a datetime when possible,
        or None if no version matches.
    """
    version = (
        _get_nssp_dataset(dataset)
        .load.resolve_version(version_spec=_version_spec(as_of))
        .version
    )
    return _version_to_datetime(version)


@overload
def get_nssp(
    disease: str | Iterable[str] | None = None,
    state_abb: str | Iterable[str] | None = None,
    dataset: NSSPDataset = "gold",
    as_of: dt.date | None = None,
    start_date: dt.date | None = None,
    end_date: dt.date | None = None,
    lazy: Literal[True] = ...,
    apply_exclusions: bool = False,
    exclusion_as_of: dt.date | None = None,
) -> pl.LazyFrame: ...


@overload
def get_nssp(
    disease: str | Iterable[str] | None = None,
    state_abb: str | Iterable[str] | None = None,
    dataset: NSSPDataset = "gold",
    as_of: dt.date | None = None,
    start_date: dt.date | None = None,
    end_date: dt.date | None = None,
    lazy: Literal[False] = ...,
    apply_exclusions: bool = False,
    exclusion_as_of: dt.date | None = None,
) -> pl.DataFrame: ...


def get_nssp(
    disease: str | Iterable[str] | None = None,
    state_abb: str | Iterable[str] | None = None,
    dataset: NSSPDataset = "gold",
    as_of: dt.date | None = None,
    start_date: dt.date | None = None,
    end_date: dt.date | None = None,
    lazy: bool = True,
    apply_exclusions: bool = False,
    exclusion_as_of: dt.date | None = None,
) -> pl.DataFrame | pl.LazyFrame:
    """
    Retrieve and filter NSSP emergency department data.

    This function retrieves vintages of NSSP emergency department
    visits data specified by the `as_of` date from the
    datacat.public.stf catalog. It filters data for a specific disease
    and location, within optional date boundaries, as available up to
    a specified reference date.

    Parameters
    ----------
    disease
        The disease to filter for ("covid", "flu", "rsv", or the aggregate
        "total" series). If None, all diseases are included.
    state_abb
        Location abbreviation to filter for. If None, all locations are included.
    dataset
        One of the two datasets to retrieve from datacat: "gold" or
        "comprehensive" (defaults to "gold").
    as_of
        Reference date for data availability. Only data available as of this date will be used.
        If None, all available data will be used (defaults to None).
    start_date
        Start date for filtering data (inclusive). If None, no lower bound is applied (defaults to None).
    end_date
        End date for filtering data (inclusive). If None, no upper bound is applied (defaults to None).
    lazy
        Whether to return a lazy frame (defaults to True). If True, returns a
        `pl.LazyFrame`; if False, returns a `pl.DataFrame`.
    apply_exclusions
        Add Boolean ``exclude`` flags from a published data-review snapshot,
        without removing observations. Defaults to False. Reviews apply to all
        diseases at a location. The latest ``report_date`` in the loaded NSSP
        data selects the review date, before applying observation filters.
    exclusion_as_of
        The latest catalog version date to consider for the data-review
        snapshot, or None for the newest available snapshot. Uses the same
        inclusive cutoff convention as ``as_of``, independently of the NSSP
        vintage and review reference date. Requires
        ``apply_exclusions=True``. Use
        ``resolve_nssp_exclusion_version(as_of=exclusion_as_of)`` to record the
        selected version.

    Returns
    -------
    pl.DataFrame | pl.LazyFrame
        Aggregated ED counts with columns:
        `date`, `state_abb`, `disease`, `target_type`, `value`, and `exclude`.
        `target_type` is always `"inc ed visits"`, and `value` is the
        corresponding incident ED-visit count.
        ``exclude`` is always a non-null Boolean column. Flags are False when
        exclusions are disabled or missing. A missing snapshot or review date
        emits a warning; an incompatible snapshot schema raises ValueError.

    Notes
    -----
    - Catalog disease labels are converted to the canonical names "covid",
      "flu", and "rsv".
    - The function only includes data from parquet files with dates up to and including the as_of date.
    """
    if exclusion_as_of is not None and not apply_exclusions:
        raise ValueError("exclusion_as_of requires apply_exclusions=True")

    state_abb = ensure_list(state_abb)
    get_all_locs = not state_abb

    disease = canonicalize_diseases(disease)
    get_all_diseases = not disease

    datacat_dataset = _get_nssp_dataset(dataset)
    dat = datacat_dataset.load.get_dataframe(
        output="pl_lazy", version_spec=_version_spec(as_of)
    )
    reference_date = (
        dat.select(pl.col("report_date").max()).collect().item()
        if apply_exclusions
        else None
    )

    national_required = get_all_locs or "US" in state_abb

    filters = [
        pl.col("metric") == "count_ed_visits",
    ]

    if not get_all_diseases:
        filters.append(pl.col("disease").is_in(disease))
    if start_date:
        filters.append(pl.col("date") >= start_date)
    if end_date:
        filters.append(pl.col("date") <= end_date)

    dat = (
        dat.rename({"reference_date": "date", "geo_value": "state_abb"})
        .with_columns(canonical_disease_expr())
        .filter(*filters)
    )

    state_locs = [loc for loc in state_abb if loc != "US"]
    state_dat = (
        dat if get_all_locs else dat.filter(pl.col("state_abb").is_in(state_locs))
    )

    combined_dat = (
        pl.concat(
            [
                state_dat.with_columns(pl.col("state_abb").cast(pl.String)),
                dat.with_columns(pl.lit("US").alias("state_abb")),
            ]
        )
        if national_required
        else state_dat
    )

    result = (
        combined_dat.group_by("date", "disease", "state_abb")
        .agg(pl.col("value").sum())
        .with_columns(pl.lit("inc ed visits").alias("target_type"))
        .select("date", "state_abb", "disease", "target_type", "value")
        .sort("state_abb", "disease", "date")
    )

    if not get_all_diseases:
        result_disease = (
            result.unique("disease").collect().get_column("disease").to_list()
        )
        if missing_diseases := set(disease) - set(result_disease):
            warnings.warn(
                f"Requested diseases {missing_diseases} not found in results."
            )

    if not get_all_locs:
        result_loc_abbr = (
            result.unique("state_abb").collect().get_column("state_abb").to_list()
        )
        if missing_locs := set(state_abb) - set(result_loc_abbr):
            warnings.warn(f"Requested locations {missing_locs} not found in results.")

    review_flags = (
        _review_flags(reference_date, exclusion_as_of)
        if reference_date is not None
        else None
    )
    if review_flags is not None:
        review_flags = review_flags.with_columns(
            pl.col("state_abb").cast(result.collect_schema()["state_abb"])
        )
        result = result.join(
            review_flags.lazy(),
            on=["state_abb", "date"],
            how="left",
            validate="m:1",
            maintain_order="left",
        ).with_columns(pl.col("exclude").fill_null(False))
    else:
        result = result.with_columns(pl.lit(False).alias("exclude"))

    if not lazy:
        result = result.collect()
    return result
