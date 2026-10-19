import datetime as dt
from types import SimpleNamespace

import polars as pl
import pytest
from cfa.catalog.public.datasets.stf.schemas.nssp_exclusions import load_mock_data
from cfa.dataops.utils import version_matcher
from polars.testing import assert_frame_equal

import cfa.stf.data.nssp as nssp
from cfa.stf.data import ensure_list, resolve_nssp_exclusion_version
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
    ).with_columns(
        pl.col("geo_value").cast(pl.Categorical),
        pl.lit(dt.date(2024, 1, 14)).alias("report_date"),
    )


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
    assert result.columns == [
        "date",
        "state_abb",
        "disease",
        "target_type",
        "value",
        "exclude",
    ]
    assert _unique_values(result, "target_type") == {"inc ed visits"}


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
    assert result.columns == [
        "date",
        "state_abb",
        "disease",
        "target_type",
        "value",
        "exclude",
    ]
    assert _unique_values(result, "target_type") == {"inc ed visits"}


@requires_ext_catalog
@pytest.mark.parametrize("dataset", ["gold", "comprehensive"])
def test_catalog_resolve_nssp_version(dataset) -> None:
    result = nssp.resolve_nssp_version(dataset=dataset)

    assert isinstance(result, dt.datetime)


REVIEW_DATE = dt.date(2024, 1, 14)


@pytest.fixture
def reviewed_exclusions(nssp_data) -> pl.DataFrame:
    locations = sorted(set(nssp_data.get_column("geo_value")) | {"US"})
    return pl.DataFrame(
        {
            "report_date": [REVIEW_DATE] * len(locations),
            "state_abb": locations,
            "excluded_dates": [
                [dt.date(2024, 1, 13)] if loc == "CA" else [] for loc in locations
            ],
        },
        schema={
            "report_date": pl.Date,
            "state_abb": pl.String,
            "excluded_dates": pl.List(pl.Date),
        },
    )


@pytest.fixture
def exclusion_catalog(monkeypatch, reviewed_exclusions):
    """Record catalog requests without using Azure credentials."""
    state = SimpleNamespace(
        frame=reviewed_exclusions,
        versions=["2024-01-14T12-00-00", "2024-01-15T12-00-00"],
        loads=[],
        resolutions=[],
    )

    def resolve_version(*, version_spec):
        state.resolutions.append(version_spec)
        return SimpleNamespace(version=version_matcher(version_spec, state.versions))

    def get_dataframe(*, output, version_spec):
        assert output == "polars"
        state.loads.append(version_spec)
        return state.frame

    endpoint = nssp.datacat.public.stf.nssp_exclusions.load
    monkeypatch.setattr(endpoint, "resolve_version", resolve_version)
    monkeypatch.setattr(endpoint, "get_dataframe", get_dataframe)
    return state


def reviewed_nssp(**kwargs):
    return nssp.get_nssp(
        **({"as_of": REVIEW_DATE, "lazy": False} | kwargs),
        get_exclusions=True,
    )


def test_review_flags_accept_catalog_schema(exclusion_catalog):
    exclusion_catalog.frame = load_mock_data(output="polars", size=2)
    report_date = exclusion_catalog.frame["report_date"][0]
    result = nssp._review_flags(report_date=report_date, exclusion_as_of=None)

    assert_frame_equal(
        result,
        pl.DataFrame(
            {
                "state_abb": ["AL"],
                "date": [report_date - dt.timedelta(days=1)],
                "exclude": [True],
            }
        ),
    )


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("dataset", ["gold", "comprehensive"])
@pytest.mark.parametrize("empty", [False, True])
def test_disabled_exclusions_return_false_without_catalog_access(
    exclusion_catalog, lazy, dataset, empty
):
    kwargs = {"lazy": lazy, "dataset": dataset}
    if empty:
        kwargs["start_date"] = dt.date(2025, 1, 1)
    default = nssp.get_nssp(**kwargs)
    explicit = nssp.get_nssp(**kwargs, get_exclusions=False)
    assert isinstance(default, pl.LazyFrame if lazy else pl.DataFrame)
    if lazy:
        default, explicit = default.collect(), explicit.collect()
    assert_frame_equal(default, explicit)
    assert default.is_empty() == empty
    assert default.schema["exclude"] == pl.Boolean
    assert default["exclude"].null_count() == 0
    assert not default["exclude"].any()
    assert exclusion_catalog.resolutions == exclusion_catalog.loads == []


def test_exclusion_as_of_requires_opt_in(monkeypatch, exclusion_catalog):
    def unexpected_load(*args, **kwargs):
        pytest.fail("Invalid options must fail before loading NSSP")

    monkeypatch.setattr(nssp, "_get_nssp_dataset", unexpected_load)
    with pytest.raises(ValueError, match="requires get_exclusions"):
        nssp.get_nssp(exclusion_as_of=dt.date(2024, 1, 15))
    assert exclusion_catalog.resolutions == exclusion_catalog.loads == []


@pytest.mark.parametrize("dataset", ["gold", "comprehensive"])
@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("state_abb", [None, "CA", ["CA", "AK"]])
def test_review_flags_preserve_observations(
    exclusion_catalog, dataset, lazy, state_abb
):
    expected = nssp.get_nssp(
        dataset=dataset, as_of=REVIEW_DATE, state_abb=state_abb, lazy=False
    )
    result = reviewed_nssp(dataset=dataset, lazy=lazy, state_abb=state_abb)
    assert isinstance(result, pl.LazyFrame if lazy else pl.DataFrame)
    if lazy:
        result = result.collect()
    assert_frame_equal(result.drop("exclude"), expected.drop("exclude"))
    assert result.schema["exclude"] == pl.Boolean
    assert result["exclude"].null_count() == 0
    assert result.filter("exclude").select("state_abb", "disease").rows() == [
        ("CA", "covid"),
        ("CA", "flu"),
    ]
    assert not result.filter(pl.col("state_abb") == "US")["exclude"].any()
    assert exclusion_catalog.loads == [max(exclusion_catalog.versions)]


def test_us_review_flags_are_independent(exclusion_catalog):
    exclusion_catalog.frame = exclusion_catalog.frame.with_columns(
        pl.when(pl.col("state_abb") == "US")
        .then(pl.lit([dt.date(2024, 1, 6)], dtype=pl.List(pl.Date)))
        .otherwise(pl.col("excluded_dates"))
        .alias("excluded_dates")
    )
    result = reviewed_nssp(state_abb=["CA", "US"], disease=["flu", "covid"])
    assert result.filter("exclude").select("state_abb", "date").unique().sort(
        "state_abb"
    ).rows() == [("CA", dt.date(2024, 1, 13)), ("US", dt.date(2024, 1, 6))]


@pytest.mark.parametrize("empty", [False, True])
def test_review_with_date_filters(exclusion_catalog, empty):
    start = dt.date(2025, 1, 1) if empty else dt.date(2024, 1, 13)
    kwargs = {"start_date": start, "end_date": start}
    expected = nssp.get_nssp(**kwargs, lazy=False)
    result = reviewed_nssp(**kwargs)
    assert_frame_equal(result.drop("exclude"), expected.drop("exclude"))
    assert result.is_empty() == empty


@pytest.mark.parametrize(
    "cutoff,spec,expected",
    [
        (None, "<=9999-12-31T00-00-00", "2024-01-15T12-00-00"),
        (dt.date(2024, 1, 15), "<=2024-01-15T00-00-00", "2024-01-14T12-00-00"),
        (dt.date(2024, 1, 16), "<=2024-01-16T00-00-00", "2024-01-15T12-00-00"),
        (
            dt.datetime(2024, 1, 15, 11, 59, 59),
            "<=2024-01-15T11-59-59",
            "2024-01-14T12-00-00",
        ),
        (dt.datetime(2024, 1, 15, 12), "<=2024-01-15T12-00-00", "2024-01-15T12-00-00"),
    ],
)
def test_exclusion_as_of_selects_latest_at_cutoff(
    exclusion_catalog, cutoff, spec, expected
):
    resolved = resolve_nssp_exclusion_version(as_of=cutoff)
    assert resolved == dt.datetime.strptime(expected, "%Y-%m-%dT%H-%M-%S")
    result = reviewed_nssp(exclusion_as_of=cutoff)
    assert result["exclude"].any()
    assert exclusion_catalog.resolutions == [spec, spec]
    assert exclusion_catalog.loads == [expected]


def test_exclusion_resolver_preserves_unparsable_version(
    monkeypatch, exclusion_catalog
):
    version = "release-review-final"
    monkeypatch.setattr(
        nssp.datacat.public.stf.nssp_exclusions.load,
        "resolve_version",
        lambda **_: SimpleNamespace(version=version),
    )
    assert resolve_nssp_exclusion_version() == version
    assert exclusion_catalog.loads == []


@pytest.mark.parametrize("empty_catalog", [False, True])
@pytest.mark.parametrize("lazy", [False, True])
def test_no_exclusions_match_cutoff(exclusion_catalog, empty_catalog, lazy):
    cutoff = dt.date(2024, 1, 14)
    if empty_catalog:
        exclusion_catalog.versions = []
        cutoff = None
    assert resolve_nssp_exclusion_version(as_of=cutoff) is None
    with pytest.warns(UserWarning, match="No published.*returning exclude=False"):
        result = reviewed_nssp(exclusion_as_of=cutoff, lazy=lazy)
    if lazy:
        result = result.collect()
    assert_frame_equal(result, nssp.get_nssp(as_of=REVIEW_DATE, lazy=False))
    assert exclusion_catalog.loads == []


@pytest.mark.parametrize("dataset", ["gold", "comprehensive"])
@pytest.mark.parametrize("as_of", [None, dt.date(2024, 1, 15)])
def test_review_date_comes_from_loaded_nssp(
    monkeypatch, exclusion_catalog, nssp_data, dataset, as_of
):
    calls = []
    endpoint = nssp._get_nssp_dataset(dataset).load

    def resolve_version(**kwargs):
        pytest.fail("Review dates should come from the loaded data")

    def get_dataframe(*, output, version_spec):
        calls.append(("load", version_spec))
        return nssp_data.lazy()

    monkeypatch.setattr(endpoint, "resolve_version", resolve_version)
    monkeypatch.setattr(endpoint, "get_dataframe", get_dataframe)
    assert reviewed_nssp(as_of=as_of, dataset=dataset)["exclude"].any()
    expected_spec = (
        "<=9999-12-31T00-00-00" if as_of is None else "<=2024-01-15T00-00-00"
    )
    assert calls == [("load", expected_spec)]


def test_latest_report_date_is_selected_before_filters(
    monkeypatch, exclusion_catalog, nssp_data
):
    data = nssp_data.with_columns(
        pl.when(pl.col("metric") == "other_metric")
        .then(pl.lit(REVIEW_DATE))
        .otherwise(pl.lit(dt.date(2024, 1, 13)))
        .alias("report_date")
    )
    monkeypatch.setattr(
        nssp._get_nssp_dataset("comprehensive").load,
        "get_dataframe",
        lazy_catalog_loader(data),
    )
    assert reviewed_nssp(dataset="comprehensive", state_abb="CA")["exclude"].any()


def test_empty_nssp_needs_no_review_date(monkeypatch, exclusion_catalog, nssp_data):
    monkeypatch.setattr(
        nssp._get_nssp_dataset("gold").load,
        "get_dataframe",
        lazy_catalog_loader(nssp_data.clear()),
    )
    result = reviewed_nssp(as_of=None)
    assert result.is_empty()
    assert result.schema["exclude"] == pl.Boolean
    assert exclusion_catalog.resolutions == exclusion_catalog.loads == []


@pytest.mark.parametrize("empty_snapshot", [False, True])
def test_missing_review_warns_and_flags_false(exclusion_catalog, empty_snapshot):
    if empty_snapshot:
        exclusion_catalog.frame = exclusion_catalog.frame.clear()
    else:
        exclusion_catalog.frame = exclusion_catalog.frame.with_columns(
            pl.lit(dt.date(2024, 1, 16)).alias("report_date")
        )
    with pytest.warns(UserWarning, match="2024-01-15T12-00-00.*2024-01-14"):
        result = reviewed_nssp()
    assert not result["exclude"].any()


def test_reviewed_empty_lists_do_not_warn(exclusion_catalog, recwarn):
    exclusion_catalog.frame = exclusion_catalog.frame.with_columns(
        pl.lit([], dtype=pl.List(pl.Date)).alias("excluded_dates")
    )
    assert not reviewed_nssp()["exclude"].any()
    assert not recwarn


@pytest.mark.parametrize(
    "malformation,match",
    [
        (lambda f: f.drop("excluded_dates"), "schema"),
        (lambda f: f.with_columns(extra=pl.lit(1)), "schema"),
        (lambda f: f.with_columns(pl.col("report_date").cast(pl.String)), "schema"),
        (
            lambda f: f.with_columns(pl.col("excluded_dates").cast(pl.List(pl.String))),
            "schema",
        ),
    ],
)
def test_malformed_snapshot_schema_rejected(exclusion_catalog, malformation, match):
    exclusion_catalog.frame = malformation(exclusion_catalog.frame)
    with pytest.raises(ValueError, match=match):
        reviewed_nssp(state_abb="CA")


def test_unrelated_malformed_review_is_ignored(exclusion_catalog):
    malformed = exclusion_catalog.frame.head(1).with_columns(
        report_date=pl.lit(dt.date(2024, 1, 1)),
        state_abb=pl.lit(None, dtype=pl.String),
        excluded_dates=pl.lit([None], dtype=pl.List(pl.Date)),
    )
    exclusion_catalog.frame = pl.concat([exclusion_catalog.frame, malformed, malformed])
    assert reviewed_nssp()["exclude"].any()


def test_catalog_errors_propagate(monkeypatch, exclusion_catalog):
    def broken_load(*args, **kwargs):
        raise OSError("Azure unavailable")

    monkeypatch.setattr(
        nssp.datacat.public.stf.nssp_exclusions.load, "get_dataframe", broken_load
    )
    with pytest.raises(OSError, match="Azure unavailable"):
        reviewed_nssp()


@pytest.mark.parametrize("reviewed", [False, True])
def test_arbitrary_nssp_location_uses_available_review(
    monkeypatch, exclusion_catalog, nssp_data, reviewed
):
    data = nssp_data.with_columns(pl.lit("PR").alias("geo_value"))
    monkeypatch.setattr(
        nssp._get_nssp_dataset("gold").load, "get_dataframe", lazy_catalog_loader(data)
    )
    if reviewed:
        exclusion_catalog.frame = exclusion_catalog.frame.with_columns(
            pl.col("state_abb").replace({"CA": "PR"})
        )
        result = reviewed_nssp(state_abb="PR")
        assert result.height > 0
        assert result["exclude"].any()
    else:
        result = reviewed_nssp(state_abb="PR")
        assert result.height > 0
        assert not result["exclude"].any()


@pytest.mark.parametrize("lazy", [False, True])
@pytest.mark.parametrize("state_abb", [None, "CA", ["CA", "US"]])
def test_missing_location_review_defaults_to_false(exclusion_catalog, lazy, state_abb):
    exclusion_catalog.frame = exclusion_catalog.frame.filter(
        pl.col("state_abb") != "CA"
    )
    exclusion_catalog.frame = exclusion_catalog.frame.with_columns(
        pl.when(pl.col("state_abb") == "US")
        .then(pl.lit([dt.date(2024, 1, 13)], dtype=pl.List(pl.Date)))
        .otherwise(pl.col("excluded_dates"))
        .alias("excluded_dates")
    )
    result = reviewed_nssp(state_abb=state_abb, lazy=lazy)
    if lazy:
        result = result.collect()
    expected = nssp.get_nssp(state_abb=state_abb, as_of=REVIEW_DATE, lazy=False)
    assert_frame_equal(result.drop("exclude"), expected.drop("exclude"))
    assert result.schema["exclude"] == pl.Boolean
    assert result["exclude"].null_count() == 0
    assert not result.filter(pl.col("state_abb") == "CA")["exclude"].any()
    if state_abb != "CA":
        assert result.filter(pl.col("state_abb") == "US")["exclude"].any()


def test_partial_review_respects_nssp_filters(exclusion_catalog):
    exclusion_catalog.frame = exclusion_catalog.frame.filter(
        pl.col("state_abb") == "CA"
    )
    assert reviewed_nssp(state_abb="CA")["exclude"].any()
    # Date filtering can remove all observations.
    assert reviewed_nssp(start_date=dt.date(2025, 1, 1)).is_empty()
    # AK has no observations after this date.
    with pytest.warns(UserWarning, match="Requested locations.*AK.*not found"):
        result = reviewed_nssp(state_abb=["AK", "CA"], start_date=dt.date(2024, 1, 13))
    assert result["exclude"].all()


def test_review_locations_absent_from_nssp_are_allowed(exclusion_catalog):
    extra = exclusion_catalog.frame.head(1).with_columns(
        pl.lit("ZZ").alias("state_abb")
    )
    exclusion_catalog.frame = pl.concat([exclusion_catalog.frame, extra])
    assert reviewed_nssp()["exclude"].any()
