"""Tests for the local Fabric ``display`` stand-in."""

import importlib
import sys
from collections import namedtuple
from unittest.mock import MagicMock

import pandas as pd
import pytest
from pyspark.sql import Row

from binaryrain_fabric_local_pyspark import display as display_module
from binaryrain_fabric_local_pyspark.display import display
from binaryrain_fabric_local_pyspark.spark import SparkSession


@pytest.fixture
def render_mock(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Capture what display passes to the renderer."""
    mock = MagicMock()
    monkeypatch.setattr(display_module, "_render", mock)
    return mock


def _rendered(render_mock: MagicMock) -> object:
    """Return the single object passed to the renderer."""
    render_mock.assert_called_once()
    return render_mock.call_args.args[0]


def test_pandas_dataframe_is_rendered_unchanged(render_mock: MagicMock) -> None:
    frame = pd.DataFrame({"a": [1, 2]})
    display(frame)
    assert _rendered(render_mock) is frame


def test_pandas_summary_renders_describe(render_mock: MagicMock) -> None:
    display(pd.DataFrame({"a": [1, 2, 3]}), summary=True)
    rendered = _rendered(render_mock)
    assert "mean" in rendered.index
    assert rendered.loc["mean", "a"] == 2


def test_list_of_named_tuples_becomes_dataframe(render_mock: MagicMock) -> None:
    point = namedtuple("Point", ["x", "y"])
    display([point(1, 2), point(3, 4)])
    rendered = _rendered(render_mock)
    assert list(rendered.columns) == ["x", "y"]
    assert rendered["x"].tolist() == [1, 3]


def test_list_of_rows_becomes_dataframe(render_mock: MagicMock) -> None:
    display([Row(a=1, b="x"), Row(a=2, b="y")])
    rendered = _rendered(render_mock)
    assert list(rendered.columns) == ["a", "b"]


def test_unconvertible_list_is_rendered_as_is(render_mock: MagicMock) -> None:
    data = [{"a": 1}, 5]
    display(data)
    assert _rendered(render_mock) is data


def test_other_objects_are_passed_through(render_mock: MagicMock) -> None:
    widget = object()
    display(widget)
    assert _rendered(render_mock) is widget


def test_spark_dataframe_is_limited(
    spark: SparkSession, render_mock: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(display_module, "MAX_DISPLAY_ROWS", 5)
    display(spark.range(20))
    rendered = _rendered(render_mock)
    assert isinstance(rendered, pd.DataFrame)
    assert rendered["id"].tolist() == [0, 1, 2, 3, 4]


def test_spark_summary_renders_statistics(spark: SparkSession, render_mock: MagicMock) -> None:
    display(spark.range(1, 4), summary=True)
    rendered = _rendered(render_mock)
    assert "mean" in rendered["summary"].tolist()


def test_falls_back_to_print_without_ipython(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(display_module, "ipython_display", None)
    display(pd.DataFrame({"col": ["value"]}))
    assert "value" in capsys.readouterr().out


def test_falls_back_to_print_for_other_objects(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(display_module, "ipython_display", None)
    display(42)
    assert capsys.readouterr().out.strip() == "42"


def test_renders_with_ipython_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    ipython_display = MagicMock()
    monkeypatch.setattr(display_module, "ipython_display", ipython_display)
    frame = pd.DataFrame({"col": [1]})
    display(frame)
    ipython_display.assert_called_once_with(frame)


def test_module_import_without_ipython() -> None:
    try:
        with pytest.MonkeyPatch.context() as patch:
            patch.setitem(sys.modules, "IPython.display", None)
            importlib.reload(display_module)
            assert display_module.ipython_display is None
    finally:
        importlib.reload(display_module)
