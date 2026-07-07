from importlib.resources import files
from io import StringIO
from typing import Literal, overload

import polars as pl

_LOCATION_POPULATION_CSV = "data/us_location_population.csv"
_LOCATION_POPULATION_SCHEMA = {
    "code": pl.String,
    "abbr": pl.String,
    "hrd": pl.String,
    "name": pl.String,
    "population": pl.Int64,
}


@overload
def get_us_location_population(lazy: Literal[False] = ...) -> pl.DataFrame: ...


@overload
def get_us_location_population(lazy: Literal[True]) -> pl.LazyFrame: ...


def get_us_location_population(lazy: bool = False) -> pl.DataFrame | pl.LazyFrame:
    """
    Return US jurisdiction location codes and Census population estimates.

    The packaged table contains the United States, states, DC, and territories.
    Population values come from the Census Population Estimates Program table
    previously packaged by the R ``forecasttools`` package as
    ``us_location_pop``. Some territories do not have population estimates in
    that source and therefore have null values.
    """
    csv_text = files(__package__).joinpath(_LOCATION_POPULATION_CSV).read_text()
    data = pl.read_csv(
        StringIO(csv_text),
        schema_overrides=_LOCATION_POPULATION_SCHEMA,
        null_values=[""],
    )
    return data.lazy() if lazy else data


def get_us_loc_pop_tbl() -> pl.DataFrame:
    """
    Compatibility wrapper for the old ``cfa.stf.forecasttools`` helper.
    """
    return get_us_location_population()
