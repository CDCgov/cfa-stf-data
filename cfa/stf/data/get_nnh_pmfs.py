import datetime as dt
from typing import Literal, overload

import polars as pl
from cfa.dataops import datacat

from ._utils import (
    canonical_disease_expr,
    canonicalize_disease,
    exact_catalog_version_spec,
)


def resolve_nnh_parameter_estimates_version() -> str | None:
    """Return the parameter catalog's exact opaque version string."""
    version = datacat.public.stf.param_estimates.load.resolve_version().version
    return version


def resolve_nnh_generation_interval_pmf_version() -> str | None:
    """Resolve the parameter-estimates version used for the generation PMF."""
    return resolve_nnh_parameter_estimates_version()


def resolve_nnh_delay_pmf_version() -> str | None:
    """Resolve the parameter-estimates version used for the delay PMF."""
    return resolve_nnh_parameter_estimates_version()


def resolve_nnh_right_truncation_pmf_version() -> str | None:
    """Resolve the parameter-estimates version used for right truncation."""
    return resolve_nnh_parameter_estimates_version()


def _extract_pmf(
    df: pl.DataFrame | pl.LazyFrame,
    parameter_name: str,
) -> list[float]:
    pmf_df = df.filter(pl.col("parameter") == parameter_name)

    if isinstance(pmf_df, pl.LazyFrame):
        pmf_df = pmf_df.collect()

    if pmf_df.height != 1:
        raise ValueError(
            f"Expected exactly one {parameter_name!r} parameter row, "
            f"but found {pmf_df.height}. "
            f"Rows={pmf_df.to_dicts()}"
        )

    return pmf_df.item(0, "value").to_list()


@overload
def _filter_param_estimates(
    disease: str,
    as_of: dt.date | None = ...,
    catalog_version: str | None = ...,
    lazy: Literal[True] = ...,
) -> pl.LazyFrame: ...


@overload
def _filter_param_estimates(
    disease: str,
    as_of: dt.date | None = ...,
    catalog_version: str | None = ...,
    lazy: Literal[False] = ...,
) -> pl.DataFrame: ...


def _filter_param_estimates(
    disease: str,
    as_of: dt.date | None = None,
    catalog_version: str | None = None,
    lazy: bool = True,
) -> pl.DataFrame | pl.LazyFrame:
    source = _load_param_estimates(
        catalog_version=catalog_version,
        lazy=lazy,
    )
    return _filter_effective_param_estimates(source, disease=disease, as_of=as_of)


@overload
def _load_param_estimates(
    *, catalog_version: str | None = ..., lazy: Literal[True] = ...
) -> pl.LazyFrame: ...


@overload
def _load_param_estimates(
    *, catalog_version: str | None = ..., lazy: Literal[False] = ...
) -> pl.DataFrame: ...


def _load_param_estimates(
    *, catalog_version: str | None = None, lazy: bool = True
) -> pl.DataFrame | pl.LazyFrame:
    """Load one exact parameter catalog snapshot."""
    output = "pl_lazy" if lazy else "pl"
    return datacat.public.stf.param_estimates.load.get_dataframe(
        output=output,
        version_spec=exact_catalog_version_spec(catalog_version),
    )


def _filter_effective_param_estimates(
    source: pl.DataFrame | pl.LazyFrame,
    *,
    disease: str,
    as_of: dt.date | None,
) -> pl.DataFrame | pl.LazyFrame:
    """Select parameter rows effective for one disease and date."""
    disease = canonicalize_disease(disease)
    as_of = as_of or dt.date.max - dt.timedelta(days=1)
    result = (
        source
        .with_columns(
            canonical_disease_expr(),
            pl.col("start_date").fill_null(dt.date.min),
            pl.col("end_date").fill_null(dt.date.max),
        )
        .filter(pl.col("disease") == disease)
        .filter(
            pl.col("start_date") <= as_of,
            as_of < pl.col("end_date"),
        )
    )
    return result


def get_nnh_generation_interval_pmf(
    disease: str,
    as_of: dt.date | None = None,
    catalog_version: str | None = None,
) -> list[float]:
    """
    Filter and extract the generation interval probability mass function (PMF)
    based on disease and date filters.

    This function retrieves epidemiological parameters from
    datacat.public.stf catalog and returns the generation interval PMF.

    Parameters
    ----------
    disease
        The canonical disease name to filter for ("covid", "flu", or "rsv").
    as_of
        The date for which parameters should be valid. Parameters must have
        start_date <= as_of < end_date. Defaults to latest estimates.
    catalog_version
        Exact opaque parameter-catalog version to read.

    Returns
    -------
    list[float]
        A list representing the generation interval distribution.

    Raises
    ------
    ValueError
        If exactly one generation_interval row is not found.
    """
    dat_filtered = _filter_param_estimates(
        disease=disease,
        as_of=as_of,
        catalog_version=catalog_version,
    )
    return _extract_pmf(dat_filtered, "generation_interval")


def get_nnh_delay_pmf(
    disease: str,
    as_of: dt.date | None = None,
    catalog_version: str | None = None,
) -> list[float]:
    """
    Filter and extract the delay probability mass function (PMF)
    based on disease and date filters.

    This function retrieves epidemiological parameters from
    datacat.public.stf catalog and returns the delay PMF.

    Parameters
    ----------
    disease
        The canonical disease name to filter for ("covid", "flu", or "rsv").
    as_of
        The date for which parameters should be valid. Parameters must have
        start_date <= as_of < end_date. Defaults to latest estimates.
    catalog_version
        Exact opaque parameter-catalog version to read.

    Returns
    -------
    list[float]
        A list representing the delay distribution.

    Raises
    ------
    ValueError
        If exactly one delay row is not found.
    """
    dat_filtered = _filter_param_estimates(
        disease=disease,
        as_of=as_of,
        catalog_version=catalog_version,
    )
    delay_pmf = _extract_pmf(dat_filtered, "delay")

    return delay_pmf


def get_nnh_right_truncation_pmf(
    state_abb: str,
    disease: str,
    as_of: dt.date | None = None,
    reference_date: dt.date | None = None,
    catalog_version: str | None = None,
) -> list[float]:
    """
    Filter and extract the right truncation probability mass function (PMF)
    based on disease, location, and date filters.

    This function retrieves epidemiological parameters from
    datacat.public.stf catalog and returns the right truncation PMF.

    Parameters
    ----------
    state_abb
        Location abbreviation (geo_value) used to filter right_truncation
        parameters.
    disease
        The canonical disease name to filter for ("covid", "flu", or "rsv").
    as_of
        The date for which parameters should be valid. Parameters must have
        start_date <= as_of < end_date. Defaults to latest estimates.
    reference_date
        The reference date for filtering. Defaults to as_of value.
        Selects the most recent parameter with
        reference_date <= this value.
    catalog_version
        Exact opaque parameter-catalog version to read.

    Returns
    -------
    list[float]
        A list representing the right truncation distribution.

    Raises
    ------
    ValueError
        If exactly one right_truncation row is not found when required.
    """
    as_of = _right_truncation_as_of(state_abb, as_of)

    dat_filtered = _filter_param_estimates(
        disease=disease,
        as_of=as_of,
        catalog_version=catalog_version,
    )
    return _extract_right_truncation_pmf(
        dat_filtered,
        state_abb=state_abb,
        reference_date=reference_date or as_of or dt.date.max,
    )


def get_nnh_pmfs(
    disease: str,
    state_abb: str,
    as_of: dt.date | None = None,
    catalog_version: str | None = None,
) -> dict[str, list[float]]:
    """Load one catalog snapshot and return all PMFs used by an STF task."""
    source = _load_param_estimates(
        catalog_version=catalog_version,
        lazy=False,
    )
    common = _filter_effective_param_estimates(
        source,
        disease=disease,
        as_of=as_of,
    )
    generation = _extract_pmf(common, "generation_interval")
    delay = _extract_pmf(common, "delay")
    right_as_of = _right_truncation_as_of(state_abb, as_of)
    right_rows = _filter_effective_param_estimates(
        source,
        disease=disease,
        as_of=right_as_of,
    )
    right_truncation = _extract_right_truncation_pmf(
        right_rows,
        state_abb=state_abb,
        reference_date=right_as_of or dt.date.max,
    )
    return {
        "generation_interval_pmf": generation,
        "hospital_delay_pmf": delay,
        "ed_delay_pmf": list(delay),
        "ed_right_truncation_pmf": right_truncation,
    }


def _right_truncation_as_of(
    state_abb: str, as_of: dt.date | None
) -> dt.date | None:
    """Apply the catalog's last supported effective date for Georgia."""
    if state_abb == "GA" and (as_of is None or as_of > dt.date(2025, 10, 14)):
        return dt.date(2025, 10, 14)
    return as_of


def _extract_right_truncation_pmf(
    dat_filtered: pl.DataFrame | pl.LazyFrame,
    *,
    state_abb: str,
    reference_date: dt.date,
) -> list[float]:
    """Select the latest location-specific right-truncation parameter row."""

    right_truncation_df = (
        dat_filtered.filter(pl.col("geo_value") == state_abb)
        .filter(pl.col("reference_date") <= reference_date)
        .filter(pl.col("reference_date") == pl.col("reference_date").max())
    )
    right_truncation_pmf = _extract_pmf(right_truncation_df, "right_truncation")

    return right_truncation_pmf
