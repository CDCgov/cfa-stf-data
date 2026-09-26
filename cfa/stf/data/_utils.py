import datetime as dt
from collections.abc import Iterable
from typing import overload

import polars as pl

CANONICAL_DISEASES = ("covid", "flu", "rsv")

CATALOG_DISEASE_TO_CANONICAL = {
    "covid": "covid",
    "flu": "flu",
    "rsv": "rsv",
    "total": "total",
    "COVID-19": "covid",
    "COVID-19/Omicron": "covid",
    "Influenza": "flu",
    "RSV": "rsv",
    "Total": "total",
}


def canonical_disease_expr() -> pl.Expr:
    """Map catalog disease labels to the package's canonical names."""
    return (
        pl.col("disease").cast(pl.String).replace_strict(CATALOG_DISEASE_TO_CANONICAL)
    )


def canonicalize_disease(disease: str) -> str:
    """Convert a disease label to its canonical name when a mapping exists."""
    return CATALOG_DISEASE_TO_CANONICAL.get(disease, disease)


def canonicalize_diseases(
    disease: str | Iterable[str] | None,
) -> list[str]:
    """Convert one or more disease labels to canonical names."""
    return [canonicalize_disease(value) for value in ensure_list(disease)]


def _version_spec(as_of: dt.date | None) -> str:
    as_of = as_of or dt.date.max
    return f"<={as_of.strftime('%Y-%m-%dT%H-%M-%S')}"


def catalog_version_spec(
    *, as_of: dt.date | None = None, catalog_version: str | None = None
) -> str:
    """Return a catalog constraint for either an as-of date or exact version."""
    if as_of is not None and catalog_version is not None:
        raise ValueError("as_of and catalog_version cannot both select a catalog version")
    if catalog_version is not None:
        if not catalog_version:
            raise ValueError("catalog_version must be a non-empty string")
        return f"=={catalog_version}"
    return _version_spec(as_of)


def exact_catalog_version_spec(catalog_version: str | None) -> str | None:
    """Return an exact catalog constraint without interpreting effective dates."""
    if catalog_version is None:
        return None
    if not catalog_version:
        raise ValueError("catalog_version must be a non-empty string")
    return f"=={catalog_version}"


@overload
def ensure_list(x: None) -> list[None]: ...


@overload
def ensure_list[T](x: T | Iterable[T]) -> list[T]: ...


def ensure_list[T](x: T | Iterable[T] | None) -> list[T] | list[None]:
    if x is None:
        return []
    elif isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
        return list(x)
    else:
        return [x]
