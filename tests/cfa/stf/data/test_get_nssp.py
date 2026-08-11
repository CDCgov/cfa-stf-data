import datetime as dt

import polars as pl
import pytest

import cfa.stf.data.nssp as nssp
from cfa.stf.data import ensure_list
from tests.cfa.stf.data.data_test_utils import (
    _unique_values,
    lazy_catalog_loader,
    requires_ext_catalog,
    uses_catalog,
)


@pytest.fixture
def nssp_data() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "metric": [
                "count_ed_visits",
                "count_ed_visits",
                "count_ed_visits",
                "count_ed_visits",
                "count_ed_visits",
                "count_ed_visits",
                "count_ed_visits",
                "count_ed_visits",
                "count_ed_visits",
                "count_ed_visits",
                "other_metric",
            ],
            "disease": [
                "COVID-19/Omicron",
                "Influenza",
                "RSV",
                "Total",
                "COVID-19",
                "Influenza",
                "COVID-19",
                "Influenza",
                "RSV",
                "Total",
                "COVID-19",
            ],
            "geo_value": [
                "AK",
                "AK",
                "CA",
                "CA",
                "CA",
                "CA",
                "SD",
                "SD",
                "US",
                "US",
                "CA",
            ],
            "reference_date": [
                dt.date(2024, 1, 6),
                dt.date(2024, 1, 6),
                dt.date(2024, 1, 6),
                dt.date(2024, 1, 6),
                dt.date(2024, 1, 13),
                dt.date(2024, 1, 13),
                dt.date(2024, 1, 13),
                dt.date(2024, 1, 13),
                dt.date(2024, 1, 13),
                dt.date(2024, 1, 13),
                dt.date(2024, 1, 13),
            ],
            "value": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 999],
        }
    ).with_columns(pl.col("geo_value").cast(pl.Categorical))


@pytest.fixture
def nssp_data_comprehensive(nssp_data: pl.DataFrame) -> pl.DataFrame:
    return nssp_data.with_columns(pl.col("geo_value").cast(pl.String))


@pytest.fixture(autouse=True)
def mock_nssp_data(
    monkeypatch, nssp_data: pl.DataFrame, nssp_data_comprehensive: pl.DataFrame, request
) -> None:
    if uses_catalog(request):
        return

    monkeypatch.setattr(
        nssp.datacat.public.stf.nssp_gold_v1.load,
        "get_dataframe",
        lazy_catalog_loader(nssp_data),
    )
    monkeypatch.setattr(
        nssp.datacat.public.stf.comprehensive_nssp_gold.load,
        "get_dataframe",
        lazy_catalog_loader(nssp_data_comprehensive),
    )


@pytest.mark.parametrize(
    "state_abb",
    [
        "US",
        "AK",
        ["AK", "CA"],
        ["CA", "US"],
    ],
)
def test_get_nssp_filters_locations(state_abb) -> None:
    expected_state_abbs = set(ensure_list(state_abb))
    result = set(
        _unique_values(nssp.get_nssp(state_abb=state_abb, lazy=False), "state_abb")
    )
    assert result == expected_state_abbs


@pytest.mark.parametrize(
    "disease",
    [
        "covid",
        "total",
        ["covid", "flu"],
    ],
)
def test_get_nssp_filters_diseases(disease) -> None:
    expected_diseases = set(ensure_list(disease))
    result = set(_unique_values(nssp.get_nssp(disease=disease, lazy=False), "disease"))
    assert result == expected_diseases


@pytest.mark.parametrize(
    ("disease", "expected"),
    [
        ("COVID-19", "covid"),
        ("COVID-19/Omicron", "covid"),
        ("Influenza", "flu"),
        ("RSV", "rsv"),
        ("Total", "total"),
    ],
)
def test_get_nssp_normalizes_legacy_disease_input(disease, expected) -> None:
    result = nssp.get_nssp(disease=disease, lazy=False)

    assert _unique_values(result, "disease") == {expected}


def test_get_nssp_rejects_unknown_catalog_disease(
    monkeypatch, nssp_data: pl.DataFrame
) -> None:
    unknown_disease_data = nssp_data.with_columns(
        pl.when(pl.int_range(pl.len()) == 0)
        .then(pl.lit("unknown"))
        .otherwise(pl.col("disease"))
        .alias("disease")
    )
    monkeypatch.setattr(
        nssp.datacat.public.stf.nssp_gold_v1.load,
        "get_dataframe",
        lazy_catalog_loader(unknown_disease_data),
    )

    with pytest.raises(pl.exceptions.InvalidOperationError):
        nssp.get_nssp(lazy=False)


def test_get_nssp_returns_all_locations_and_diseases() -> None:
    result = nssp.get_nssp(lazy=False)

    assert {"covid", "flu", "rsv", "total"} == _unique_values(result, "disease")
    assert {"US", "CA", "SD"}.issubset(_unique_values(result, "state_abb"))
    assert result.columns == ["date", "disease", "state_abb", "value"]


def test_get_nssp_warns_about_missing_filters() -> None:
    with pytest.warns(UserWarning) as warnings:
        result = nssp.get_nssp(
            state_abb=["CA", "US", "XY"],
            disease=["covid", "flu", "ZZ"],
            lazy=False,
        )

    warning_messages = [str(warning.message) for warning in warnings]
    assert any("Requested diseases {'ZZ'} not found" in msg for msg in warning_messages)
    assert any(
        "Requested locations {'XY'} not found" in msg for msg in warning_messages
    )
    assert _unique_values(result, "state_abb") == {"CA", "US"}
    assert _unique_values(result, "disease") == {"covid", "flu"}


@pytest.mark.parametrize(
    "state_abb",
    [
        "US",
        "AK",
        ["AK", "CA"],
        ["CA", "US"],
    ],
)
def test_get_nssp_comprehensive_filters_locations(state_abb) -> None:
    expected_state_abbs = set(ensure_list(state_abb))
    result = set(
        _unique_values(
            nssp.get_nssp(state_abb=state_abb, dataset="comprehensive", lazy=False),
            "state_abb",
        )
    )
    assert result == expected_state_abbs


@pytest.mark.parametrize(
    "disease",
    [
        "covid",
        "total",
        ["covid", "flu"],
    ],
)
def test_get_nssp_comprehensive_filters_diseases(disease) -> None:
    expected_diseases = set(ensure_list(disease))
    result = set(
        _unique_values(
            nssp.get_nssp(disease=disease, dataset="comprehensive", lazy=False),
            "disease",
        )
    )
    assert result == expected_diseases


@requires_ext_catalog
@pytest.mark.parametrize(
    "state_abb",
    [
        "US",
        "AK",
        ["AK", "CA"],
        ["CA", "US"],
    ],
)
def test_catalog_get_nssp_filters_locations(state_abb) -> None:
    expected_state_abbs = set(ensure_list(state_abb))
    result = set(
        _unique_values(nssp.get_nssp(state_abb=state_abb, lazy=False), "state_abb")
    )
    assert result == expected_state_abbs


@requires_ext_catalog
@pytest.mark.parametrize(
    "disease",
    [
        "covid",
        "total",
        ["covid", "flu"],
    ],
)
def test_catalog_get_nssp_filters_diseases(disease) -> None:
    expected_diseases = set(ensure_list(disease))
    result = set(_unique_values(nssp.get_nssp(disease=disease, lazy=False), "disease"))
    assert result == expected_diseases


@requires_ext_catalog
def test_catalog_get_nssp_returns_all_locations_and_diseases() -> None:
    result = nssp.get_nssp(lazy=False)

    assert {"covid", "flu", "rsv", "total"} == _unique_values(result, "disease")
    assert {"US", "CA", "SD"}.issubset(_unique_values(result, "state_abb"))
    assert result.columns == ["date", "disease", "state_abb", "value"]


@requires_ext_catalog
@pytest.mark.parametrize("dataset", ["gold", "comprehensive"])
def test_catalog_resolve_nssp_version(dataset) -> None:
    result = nssp.resolve_nssp_version(dataset=dataset)

    assert isinstance(result, dt.datetime)
