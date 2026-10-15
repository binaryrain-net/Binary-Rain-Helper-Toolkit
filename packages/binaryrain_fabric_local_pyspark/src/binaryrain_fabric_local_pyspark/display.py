"""Provide a local stand-in for the Fabric notebook ``display`` function."""

import logging
from typing import Any

import pandas as pd
from pyspark.sql import DataFrame, Row

try:
    from IPython.display import display as ipython_display
except ImportError:
    ipython_display = None

logger = logging.getLogger(__name__)

# Fabric renders at most this many rows of a Spark DataFrame.
MAX_DISPLAY_ROWS = 1000


def _render(obj: Any) -> None:
    """
    Render an object in the notebook, or print it when IPython is not installed.

    :param Any obj:
        The object to render.
    """
    if ipython_display is None:
        print(obj.to_string() if isinstance(obj, pd.DataFrame) else obj)
        return
    with pd.option_context("display.max_columns", None):
        ipython_display(obj)


def _list_to_frame(data: list[Any]) -> pd.DataFrame | None:
    """
    Convert a list to a pandas DataFrame.

    :param list[Any] data:
        A list of Spark rows, named tuples, dicts, or other records.

    :returns pd.DataFrame | None:
        The converted DataFrame, or None when the list cannot be converted.
    """
    if data and all(isinstance(item, Row) for item in data):
        data = [item.asDict(recursive=True) for item in data]
    try:
        return pd.DataFrame(data)
    except (TypeError, ValueError):
        logger.debug("List could not be converted to a DataFrame.", exc_info=True)
        return None


def display(data: Any, summary: bool = False) -> None:
    """
    Display a dataset in a local notebook the way Fabric's ``display`` does.

    Spark DataFrames are limited to the first ``MAX_DISPLAY_ROWS`` rows and shown as a
    pandas table. Lists are converted to a DataFrame when possible. Any other object,
    for example an ipywidgets widget, is passed to IPython's display unchanged.

    :param Any data:
        A Spark DataFrame, pandas DataFrame, list, or any object IPython can display.
    :param bool summary:
        Show summary statistics per column instead of the rows. Default is False.
    """
    if isinstance(data, DataFrame):
        frame = data.summary() if summary else data.limit(MAX_DISPLAY_ROWS)
        # Collecting rows avoids toPandas, which warns about pandas 3 support.
        rows = [row.asDict(recursive=True) for row in frame.collect()]
        _render(pd.DataFrame(rows, columns=frame.columns))
        return

    if isinstance(data, list):
        converted = _list_to_frame(data)
        if converted is None:
            _render(data)
            return
        data = converted

    if isinstance(data, pd.DataFrame):
        _render(data.describe(include="all") if summary and not data.empty else data)
        return

    _render(data)
