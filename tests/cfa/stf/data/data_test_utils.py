from types import SimpleNamespace

import pytest
from cfa.cloudops.util import check_ext_env

CATALOG_SKIP_REASON = "requires external CFA data environment"


def ensure_mock_stf_catalog(monkeypatch, datacat, *dataset_names: str) -> None:
    public = getattr(datacat, "public", SimpleNamespace())
    monkeypatch.setattr(datacat, "public", public, raising=False)

    stf = getattr(public, "stf", SimpleNamespace())
    monkeypatch.setattr(public, "stf", stf, raising=False)

    for dataset_name in dataset_names:
        dataset = getattr(stf, dataset_name, SimpleNamespace())
        monkeypatch.setattr(stf, dataset_name, dataset, raising=False)

        load = getattr(dataset, "load", SimpleNamespace(get_dataframe=None))
        if not hasattr(load, "get_dataframe"):
            monkeypatch.setattr(load, "get_dataframe", None, raising=False)
        monkeypatch.setattr(dataset, "load", load, raising=False)


def requires_ext_catalog(test_func):
    skip_without_ext_env = pytest.mark.skipif(
        not check_ext_env(),
        reason=CATALOG_SKIP_REASON,
    )
    return skip_without_ext_env(pytest.mark.catalog(test_func))


def uses_catalog(request) -> bool:
    return request.node.get_closest_marker("catalog") is not None


def lazy_catalog_loader(df):
    def get_dataframe(output: str, version_spec: str):
        if output != "pl_lazy":
            raise ValueError(f"Unexpected output={output!r}")
        if version_spec is None:
            raise ValueError("Expected a version constraint")
        return df.lazy()

    return get_dataframe


def _unique_values(df, column: str) -> set[str]:
    if hasattr(df, "collect"):
        df = df.collect()
    return set(df.unique(column).get_column(column).to_list())
