import datetime as dt
import warnings
from collections.abc import Iterable
from typing import Literal, overload

import polars as pl
from cfa.dataops import datacat

from ._utils import _version_spec, _version_to_datetime, ensure_list


def _get_nhsn_hrd_dataset(prelim: bool):
    return datacat.public.stf.nhsn_hrd_prelim if prelim else datacat.public.stf.nhsn_hrd


def resolve_nhsn_hrd_version(
    prelim: bool = True,
    as_of: dt.date | None = None,
) -> dt.datetime | str | None:
    """Resolve the catalog version that
    [`get_nhsn_hrd`][cfa.stf.data.get_nhsn_hrd] would load.

    Parameters
    ----------
    prelim
        Whether to resolve the ``nhsn_hrd_prelim`` dataset rather than
        ``nhsn_hrd``.
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
        _get_nhsn_hrd_dataset(prelim)
        .load.resolve_version(version_spec=_version_spec(as_of))
        .version
    )
    return _version_to_datetime(version)


@overload
def get_nhsn_hrd(
    disease: str | Iterable[str] | None = None,
    loc_abb: str | Iterable[str] | None = None,
    prelim: bool = ...,
    as_of: dt.date | None = ...,
    start_date: dt.date | None = ...,
    end_date: dt.date | None = ...,
    lazy: Literal[True] = ...,
) -> pl.LazyFrame: ...


@overload
def get_nhsn_hrd(
    disease: str | Iterable[str] | None = None,
    loc_abb: str | Iterable[str] | None = None,
    prelim: bool = ...,
    as_of: dt.date | None = ...,
    start_date: dt.date | None = ...,
    end_date: dt.date | None = ...,
    lazy: Literal[False] = ...,
) -> pl.DataFrame: ...


def get_nhsn_hrd(
    disease: str | Iterable[str] | None = None,
    loc_abb: str | Iterable[str] | None = None,
    prelim: bool = True,
    as_of: dt.date | None = None,
    start_date: dt.date | None = None,
    end_date: dt.date | None = None,
    lazy: bool = True,
) -> pl.DataFrame | pl.LazyFrame:
    """
    Retrieve and filter NHSN hospital respiratory data based on specified criteria.

    This function retrieves vintages of NHSN hrd data specified by
    the `as_of` date from the datacat.public.stf catalog, applies filters
    for a specific disease, location, and dates if provided.

    Parameters
    ----------
    disease
        The disease to filter for ("COVID-19", "Influenza", or "RSV"). If None, all diseases are included.
    loc_abb
        The location abbreviation to filter for. If None, all locations are included.
    prelim
        Whether to retrieve "nhsn_hrd_prelim" data as opposed to "nhsn_hrd" data (defaults to True).
    as_of
        The reference date for filtering. If None, the most recent 'as_of' date is used.
    start_date
        The start date for the time period to include. If None, no lower bound is applied.
    end_date
        The end date for the time period to include. If None, no upper bound is applied.
    lazy
        Whether to return a lazy frame (defaults to True). If True, returns a
        `pl.LazyFrame`; if False, returns a `pl.DataFrame`.

    Returns
    -------
    pl.DataFrame | pl.LazyFrame
        Filtered data in long format with columns:
        `weekendingdate`, `jurisdiction`, `disease`, and `hospital_admissions`.
    """
    disease = ensure_list(disease)
    get_all_diseases = not disease

    loc_abb = ensure_list(loc_abb)
    get_all_locs = not loc_abb

    nhsn_disease_map = {
        "COVID-19": "totalconfc19newadm",
        "Influenza": "totalconfflunewadm",
        "RSV": "totalconfrsvnewadm",
    }

    disease_valid = (
        list(nhsn_disease_map.keys())
        if get_all_diseases
        else [x for x in disease if x in nhsn_disease_map.keys()]
    )

    raw_disease_col = [nhsn_disease_map.get(x) for x in disease_valid]

    inv_nhsn_disease_map = {nhsn_disease_map.get(x): x for x in disease_valid}

    filters = []
    if not get_all_locs:
        filters.append(pl.col("jurisdiction").is_in(loc_abb))
    if start_date:
        filters.append(pl.col("weekendingdate") >= start_date)
    if end_date:
        filters.append(pl.col("weekendingdate") <= end_date)

    datacat_dataset = _get_nhsn_hrd_dataset(prelim)

    dat = (
        datacat_dataset.load.get_dataframe(
            output="pl_lazy", version_spec=_version_spec(as_of)
        )
        .select(raw_disease_col + ["weekendingdate", "jurisdiction"])
        .with_columns(
            pl.col("jurisdiction").replace_strict(
                {"USA": "US"}, default=pl.col("jurisdiction")
            )
        )
        .filter(*filters)
        .rename(inv_nhsn_disease_map)
        .unpivot(
            on=disease_valid,
            index=["weekendingdate", "jurisdiction"],
            variable_name="disease",
            value_name="hospital_admissions",
        )
        .sort("jurisdiction", "disease", "weekendingdate")
    )

    if not get_all_diseases:
        result_disease = dat.unique("disease").collect().get_column("disease").to_list()
        if missing_diseases := set(disease) - set(result_disease):
            warnings.warn(
                f"Requested diseases {missing_diseases} not found in results."
            )
    if not get_all_locs:
        result_loc_abbr = (
            dat.unique("jurisdiction").collect().get_column("jurisdiction").to_list()
        )
        if missing_locs := set(loc_abb) - set(result_loc_abbr):
            warnings.warn(f"Requested locations {missing_locs} not found in results.")

    if not lazy:
        dat = dat.collect()
    return dat
