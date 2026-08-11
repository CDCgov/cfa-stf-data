import datetime as dt

import polars as pl
import pytest

import cfa.stf.data.nhsn as nhsn
from cfa.stf.data import ensure_list
from tests.cfa.stf.data.data_test_utils import (
    _unique_values,
    lazy_catalog_loader,
    requires_ext_catalog,
    uses_catalog,
)


@pytest.fixture
def nhsn_hrd_data() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "weekendingdate": [
                dt.date(2024, 1, 6),
                dt.date(2024, 1, 6),
                dt.date(2024, 1, 13),
                dt.date(2024, 1, 13),
            ],
            "jurisdiction": ["USA", "AK", "CA", "SD"],
            "totalconfc19newadm": [10, 20, 30, 40],
            "totalconfflunewadm": [50, 60, 70, 80],
            "totalconfrsvnewadm": [90, 100, 110, 120],
        }
    )


@pytest.fixture(autouse=True)
def mock_nhsn_hrd_data(monkeypatch, nhsn_hrd_data: pl.DataFrame, request) -> None:
    if uses_catalog(request):
        return

    get_dataframe = lazy_catalog_loader(nhsn_hrd_data)

    monkeypatch.setattr(
        nhsn.datacat.public.stf.nhsn_hrd_prelim.load,
        "get_dataframe",
        get_dataframe,
    )
    monkeypatch.setattr(
        nhsn.datacat.public.stf.nhsn_hrd.load,
        "get_dataframe",
        get_dataframe,
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
def test_get_nhsn_hrd_filters_locations(state_abb) -> None:
    expected_state_abbs = set(ensure_list(state_abb))
    result = set(
        _unique_values(nhsn.get_nhsn_hrd(state_abb=state_abb, lazy=False), "state_abb")
    )
    assert result == expected_state_abbs


@pytest.mark.parametrize(
    "disease",
    [
        "covid",
        ["covid", "flu"],
    ],
)
def test_get_nhsn_hrd_filters_diseases(disease) -> None:
    expected_diseases = set(ensure_list(disease))
    result = set(
        _unique_values(nhsn.get_nhsn_hrd(disease=disease, lazy=False), "disease")
    )
    assert result == expected_diseases


def test_get_nhsn_hrd_normalizes_legacy_disease_inputs() -> None:
    result = nhsn.get_nhsn_hrd(
        disease=["COVID-19", "Influenza", "RSV"],
        lazy=False,
    )

    assert _unique_values(result, "disease") == {"covid", "flu", "rsv"}


def test_get_nhsn_hrd_returns_all_locations_and_diseases() -> None:
    result = nhsn.get_nhsn_hrd(lazy=False)

    assert {"covid", "flu", "rsv"} == _unique_values(result, "disease")
    assert {"US", "CA", "SD"}.issubset(_unique_values(result, "state_abb"))
    assert result.columns == ["date", "state_abb", "disease", "target_type", "value"]
    assert _unique_values(result, "target_type") == {"wk inc hosp"}
    assert (
        result.filter(
            pl.col("date") == dt.date(2024, 1, 6),
            pl.col("state_abb") == "US",
            pl.col("disease") == "covid",
        ).item(0, "value")
        == 10
    )


def test_get_nhsn_hrd_warns_about_missing_filters() -> None:
    with pytest.warns(UserWarning) as warnings:
        result = nhsn.get_nhsn_hrd(
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
def test_catalog_get_nhsn_hrd_filters_locations(
    state_abb,
) -> None:
    expected_state_abbs = set(ensure_list(state_abb))
    result = set(
        _unique_values(nhsn.get_nhsn_hrd(state_abb=state_abb, lazy=False), "state_abb")
    )
    assert result == expected_state_abbs


@requires_ext_catalog
@pytest.mark.parametrize(
    "disease",
    [
        "covid",
        ["covid", "flu"],
    ],
)
def test_catalog_get_nhsn_hrd_filters_diseases(
    disease,
) -> None:
    expected_diseases = set(ensure_list(disease))
    result = set(
        _unique_values(nhsn.get_nhsn_hrd(disease=disease, lazy=False), "disease")
    )
    assert result == expected_diseases


@requires_ext_catalog
def test_catalog_get_nhsn_hrd_returns_all_locations_and_diseases() -> None:
    result = nhsn.get_nhsn_hrd(lazy=False)

    assert {"covid", "flu", "rsv"} == _unique_values(result, "disease")
    assert {"US", "CA", "SD"}.issubset(_unique_values(result, "state_abb"))
    assert result.columns == ["date", "state_abb", "disease", "target_type", "value"]
    assert _unique_values(result, "target_type") == {"wk inc hosp"}


@requires_ext_catalog
@pytest.mark.parametrize("prelim", [True, False])
def test_catalog_resolve_nhsn_hrd_version(prelim) -> None:
    result = nhsn.resolve_nhsn_hrd_version(prelim=prelim)

    assert isinstance(result, dt.datetime)
