import datetime as dt
from collections.abc import Iterable
from typing import overload


def _version_to_datetime(version: str | None) -> dt.datetime | str | None:
    if version is None:
        return None

    try:
        return dt.datetime.fromisoformat(version)
    except ValueError:
        pass

    try:
        return dt.datetime.strptime(version, "%Y-%m-%dT%H-%M-%S")
    except ValueError:
        return version


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
