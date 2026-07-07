from collections.abc import Iterable
from typing import overload


@overload
def ensure_list[T](x: None) -> None: ...


@overload
def ensure_list[T](x: T | Iterable[T]) -> list[T]: ...


def ensure_list[T](x: T | Iterable[T] | None) -> list[T] | None:
    if x is None:
        return []
    elif isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
        return list(x)
    else:
        return [x]
