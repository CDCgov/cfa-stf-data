from collections.abc import Iterable
from typing import overload


@overload
def ensure_list[T](x: None) -> list[T]: ...


@overload
def ensure_list[T](x: T | Iterable[T]) -> list[T]: ...


def ensure_list[T](x: T | Iterable[T] | None) -> list[T]:
    if x is None:
        return []
    if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
        return list(x)
    return [x]
