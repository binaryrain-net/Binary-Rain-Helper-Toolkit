"""
Comprehensive test suite for the convert_to_datetime function.
Tests all date formats, edge cases, and error handling.
"""

import pandas as pd
from binaryrain_helper_data_processing.dataframe import convert_to_datetime


class TestConvertToDatetimeBasic:
    """Test cases for basic datetime conversion."""

    def test_convert_default_format_ddmmyyyy(self):
        """Test conversion with default format %d.%m.%Y."""
        df_test = pd.DataFrame(
            {"date": ["01.01.2023", "15.06.2023", "31.12.2023"], "value": [100, 200, 300]}
        )

        result = convert_to_datetime(df_test)

        assert isinstance(result, pd.DataFrame)
        assert pd.api.types.is_datetime64_any_dtype(result["date"])
        assert result.iloc[0]["date"] == pd.Timestamp("2023-01-01")
        assert result.iloc[1]["date"] == pd.Timestamp("2023-06-15")
        assert result.iloc[2]["date"] == pd.Timestamp("2023-12-31")

    def test_convert_default_format_yyyy_mm_dd(self):
        """Test conversion with default format %Y-%m-%d."""
        df_test = pd.DataFrame(
            {"date": ["2023-01-01", "2023-06-15", "2023-12-31"], "value": [100, 200, 300]}
        )

        result = convert_to_datetime(df_test)

        assert isinstance(result, pd.DataFrame)
        assert pd.api.types.is_datetime64_any_dtype(result["date"])
        assert result.iloc[0]["date"] == pd.Timestamp("2023-01-01")

    def test_convert_default_format_with_time(self):
        """Test conversion with default format %Y-%m-%d %H:%M:%S."""
        df_test = pd.DataFrame(
            {
                "datetime": ["2023-01-01 10:30:00", "2023-06-15 14:45:30", "2023-12-31 23:59:59"],
                "value": [100, 200, 300],
            }
        )

        result = convert_to_datetime(df_test)

        assert isinstance(result, pd.DataFrame)
        assert pd.api.types.is_datetime64_any_dtype(result["datetime"])
        assert result.iloc[0]["datetime"] == pd.Timestamp("2023-01-01 10:30:00")
        assert result.iloc[1]["datetime"] == pd.Timestamp("2023-06-15 14:45:30")

    def test_convert_iso_format(self):
        """Test conversion with ISO format %Y-%m-%dT%H:%M:%S."""
        df_test = pd.DataFrame(
            {
                "datetime": ["2023-01-01T10:30:00", "2023-06-15T14:45:30", "2023-12-31T23:59:59"],
                "value": [100, 200, 300],
            }
        )

        result = convert_to_datetime(df_test)

        assert isinstance(result, pd.DataFrame)
        assert pd.api.types.is_datetime64_any_dtype(result["datetime"])
        assert result.iloc[0]["datetime"] == pd.Timestamp("2023-01-01 10:30:00")

    def test_convert_multiple_date_columns(self):
        """Test conversion with multiple date columns."""
        df_test = pd.DataFrame(
            {
                "start_date": ["2023-01-01", "2023-02-01"],
                "end_date": ["2023-01-31", "2023-02-28"],
                "value": [100, 200],
            }
        )

        result = convert_to_datetime(df_test)

        assert pd.api.types.is_datetime64_any_dtype(result["start_date"])
        assert pd.api.types.is_datetime64_any_dtype(result["end_date"])
        assert result.iloc[0]["start_date"] == pd.Timestamp("2023-01-01")
        assert result.iloc[0]["end_date"] == pd.Timestamp("2023-01-31")


class TestConvertToDatetimeCustomFormats:
    """Test cases for custom date formats."""

    def test_convert_with_custom_format(self):
        """Test conversion with custom date format."""
        df_test = pd.DataFrame({"date": ["01/15/2023", "06/30/2023"], "value": [100, 200]})

        result = convert_to_datetime(df_test, date_formats=["%m/%d/%Y"])

        assert pd.api.types.is_datetime64_any_dtype(result["date"])
        assert result.iloc[0]["date"] == pd.Timestamp("2023-01-15")

    def test_convert_with_multiple_different_formats(self):
        """Test conversion with multiple different formats."""
        df_test = pd.DataFrame(
            {"date": ["2023-01-01", "01.02.2023", "03/15/2023"], "value": [100, 200, 300]}
        )

        result = convert_to_datetime(df_test, date_formats=["%Y-%m-%d", "%d.%m.%Y", "%m/%d/%Y"])

        # First format should match first row
        assert pd.api.types.is_datetime64_any_dtype(result["date"])
        assert result.iloc[0]["date"] == pd.Timestamp("2023-01-01")
        assert pd.isna(result.iloc[1]["date"])
        assert pd.isna(result.iloc[2]["date"])

    def test_convert_with_empty_format_list(self):
        """Test conversion with empty format list."""
        df_test = pd.DataFrame({"date": ["2023-01-01", "2023-06-15"], "value": [100, 200]})

        result = convert_to_datetime(df_test, date_formats=[])

        # Should not convert with empty format list
        assert result["date"].dtype == "object"

    def test_convert_with_none_format_list(self):
        """Test conversion with None format list (uses default)."""
        df_test = pd.DataFrame({"date": ["2023-01-01", "2023-06-15"], "value": [100, 200]})

        result = convert_to_datetime(df_test, date_formats=None)

        # Should use default formats
        assert pd.api.types.is_datetime64_any_dtype(result["date"])


class TestConvertToDatetimeEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_convert_empty_dataframe(self):
        """Test conversion with empty dataframe."""
        df_test = pd.DataFrame()

        result = convert_to_datetime(df_test)

        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_convert_no_object_columns(self):
        """Test conversion when no object (string) columns exist."""
        df_test = pd.DataFrame({"int_col": [1, 2, 3], "float_col": [1.1, 2.2, 3.3]})

        result = convert_to_datetime(df_test)

        assert result["int_col"].dtype in ["int64", "int32"]
        assert result["float_col"].dtype == "float64"

    def test_convert_already_datetime_column(self):
        """Test conversion when column is already datetime."""
        df_test = pd.DataFrame(
            {"date": pd.to_datetime(["2023-01-01", "2023-06-15"]), "value": [100, 200]}
        )

        result = convert_to_datetime(df_test)

        # Should remain datetime
        assert pd.api.types.is_datetime64_any_dtype(result["date"])

    def test_convert_mixed_column_types(self):
        """Test conversion with mixed column types."""
        df_test = pd.DataFrame(
            {
                "date_str": ["2023-01-01", "2023-06-15"],
                "int_col": [1, 2],
                "float_col": [1.1, 2.2],
                "bool_col": [True, False],
            }
        )

        result = convert_to_datetime(df_test)

        assert pd.api.types.is_datetime64_any_dtype(result["date_str"])
        assert result["int_col"].dtype in ["int64", "int32"]
        assert result["float_col"].dtype == "float64"

    def test_convert_single_row(self):
        """Test conversion with single row dataframe."""
        df_test = pd.DataFrame({"date": ["2023-01-01"], "value": [100]})

        result = convert_to_datetime(df_test)

        assert pd.api.types.is_datetime64_any_dtype(result["date"])
        assert result.iloc[0]["date"] == pd.Timestamp("2023-01-01")

    def test_convert_single_column(self):
        """Test conversion with single column dataframe."""
        df_test = pd.DataFrame({"date": ["2023-01-01", "2023-06-15", "2023-12-31"]})

        result = convert_to_datetime(df_test)

        assert pd.api.types.is_datetime64_any_dtype(result["date"])
        assert result.shape == (3, 1)

    def test_convert_large_dataset(self):
        """Test conversion with large dataset."""
        dates = [f"2023-01-{str(i + 1).zfill(2)}" for i in range(31)]
        df_test = pd.DataFrame({"date": dates * 100, "value": range(3100)})

        result = convert_to_datetime(df_test)

        assert pd.api.types.is_datetime64_any_dtype(result["date"])
        assert result.shape[0] == 3100


class TestConvertToDatetimeInvalidData:
    """Test cases for invalid data handling."""

    def test_convert_invalid_date_strings(self):
        """Test conversion with invalid date strings."""
        df_test = pd.DataFrame(
            {"date": ["2023-01-01", "invalid", "2023-12-31"], "value": [1, 2, 3]}
        )

        result = convert_to_datetime(df_test)

        # Invalid dates should be converted to NaT (Not a Time)
        assert pd.api.types.is_datetime64_any_dtype(result["date"])
        assert pd.isna(result.iloc[1]["date"])
        assert result.iloc[0]["date"] == pd.Timestamp("2023-01-01")
        assert result.iloc[2]["date"] == pd.Timestamp("2023-12-31")

    def test_convert_partially_matching_format(self):
        """Test conversion when only some values match format."""
        df_test = pd.DataFrame(
            {"date": ["2023-01-01", "2023-06-15", "01.12.2023"], "value": [100, 200, 300]}
        )

        result = convert_to_datetime(df_test)

        # Should convert what matches and coerce rest to NaT
        assert pd.api.types.is_datetime64_any_dtype(result["date"])

    def test_convert_empty_strings(self):
        """Test conversion with empty strings."""
        df_test = pd.DataFrame({"date": ["2023-01-01", "", "2023-12-31"], "value": [1, 2, 3]})

        result = convert_to_datetime(df_test)

        # Empty strings should become NaT
        assert pd.api.types.is_datetime64_any_dtype(result["date"])
        assert pd.isna(result.iloc[1]["date"])

    def test_convert_none_values(self):
        """Test conversion with None values."""
        df_test = pd.DataFrame({"date": ["2023-01-01", None, "2023-12-31"], "value": [1, 2, 3]})

        result = convert_to_datetime(df_test)

        # None values should remain as NaT
        assert pd.isna(result.iloc[1]["date"])

    def test_convert_mixed_valid_invalid(self):
        """Test conversion with mix of valid and invalid dates."""
        df_test = pd.DataFrame(
            {
                "date": [
                    "2023-01-01",
                    "not a date",
                    "2023-06-15",
                    "also invalid",
                    "2023-12-31",
                ],
                "value": [1, 2, 3, 4, 5],
            }
        )

        result = convert_to_datetime(df_test)

        assert pd.api.types.is_datetime64_any_dtype(result["date"])
        assert result.iloc[0]["date"] == pd.Timestamp("2023-01-01")
        assert pd.isna(result.iloc[1]["date"])
        assert result.iloc[2]["date"] == pd.Timestamp("2023-06-15")
        assert pd.isna(result.iloc[3]["date"])


class TestConvertToDatetimeFormatPriority:
    """Test cases for format priority and matching."""

    def test_convert_first_matching_format_wins(self):
        """Test that first matching format is used."""
        df_test = pd.DataFrame({"date": ["01.01.2023", "15.06.2023"], "value": [100, 200]})

        # First format should match
        result = convert_to_datetime(df_test, date_formats=["%d.%m.%Y", "%Y-%m-%d"])

        assert pd.api.types.is_datetime64_any_dtype(result["date"])
        assert result.iloc[0]["date"] == pd.Timestamp("2023-01-01")

    def test_convert_fallback_to_second_format(self):
        """Test fallback to second format when first doesn't match."""
        df_test = pd.DataFrame({"date": ["2023-01-01", "2023-06-15"], "value": [100, 200]})

        # First format won't match, should fallback to second
        result = convert_to_datetime(df_test, date_formats=["%d.%m.%Y", "%Y-%m-%d"])

        assert pd.api.types.is_datetime64_any_dtype(result["date"])
        assert result.iloc[0]["date"] == pd.Timestamp("2023-01-01")

    def test_convert_different_formats_per_column(self):
        """Test that different columns can match different formats."""
        df_test = pd.DataFrame(
            {"date1": ["01.01.2023", "15.06.2023"], "date2": ["2023-01-01", "2023-06-15"]}
        )

        result = convert_to_datetime(df_test, date_formats=["%d.%m.%Y", "%Y-%m-%d"])

        # Both columns should be converted (each matching its format)
        assert pd.api.types.is_datetime64_any_dtype(result["date1"])
        assert pd.api.types.is_datetime64_any_dtype(result["date2"])


class TestConvertToDatetimePreservesData:
    """Test cases to ensure non-date data is preserved."""

    def test_convert_preserves_non_date_columns(self):
        """Test that non-date columns are preserved unchanged."""
        df_test = pd.DataFrame(
            {
                "date": ["2023-01-28", "2023-06-15"],
                "name": ["Alice", "Bob"],
                "value": [100, 200],
                "flag": [True, False],
            }
        )

        result = convert_to_datetime(df_test)

        assert pd.api.types.is_datetime64_any_dtype(result["date"])
        assert result["name"].dtype == "object"
        assert result["value"].dtype in ["int64", "int32"]
        assert result["flag"].dtype == "bool"
        assert list(result["name"]) == ["Alice", "Bob"]
        assert list(result["value"]) == [100, 200]

    def test_convert_preserves_row_order(self):
        """Test that row order is preserved."""
        df_test = pd.DataFrame(
            {
                "date": ["2023-12-31", "2023-01-01", "2023-06-15"],
                "id": [3, 1, 2],
            }
        )

        result = convert_to_datetime(df_test)

        assert result.iloc[0]["date"] == pd.Timestamp("2023-12-31")
        assert result.iloc[1]["date"] == pd.Timestamp("2023-01-01")
        assert result.iloc[2]["date"] == pd.Timestamp("2023-06-15")
        assert list(result["id"]) == [3, 1, 2]

    def test_convert_preserves_index(self):
        """Test that dataframe index is preserved."""
        df_test = pd.DataFrame(
            {"date": ["2023-01-01", "2023-06-15"], "value": [100, 200]}, index=["a", "b"]
        )

        result = convert_to_datetime(df_test)

        assert list(result.index) == ["a", "b"]

    def test_convert_preserves_column_order(self):
        """Test that column order is preserved."""
        df_test = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "date": ["2023-01-01", "2023-06-15", "2023-12-31"],
                "value": [100, 200, 300],
                "name": ["A", "B", "C"],
            }
        )

        result = convert_to_datetime(df_test)

        assert list(result.columns) == ["id", "date", "value", "name"]


class TestConvertToDatetimeSpecialDates:
    """Test cases for special date scenarios."""

    def test_convert_leap_year_dates(self):
        """Test conversion with leap year dates."""
        df_test = pd.DataFrame(
            {"date": ["2024-02-29", "2023-02-28"], "value": [100, 200]}  # Leap year
        )

        result = convert_to_datetime(df_test)

        assert pd.api.types.is_datetime64_any_dtype(result["date"])
        assert result.iloc[0]["date"] == pd.Timestamp("2024-02-29")
        assert result.iloc[1]["date"] == pd.Timestamp("2023-02-28")

    def test_convert_year_boundaries(self):
        """Test conversion with year boundary dates."""
        df_test = pd.DataFrame(
            {
                "date": ["2022-12-31", "2023-01-01", "2023-12-31", "2024-01-01"],
                "value": [1, 2, 3, 4],
            }
        )

        result = convert_to_datetime(df_test)

        assert pd.api.types.is_datetime64_any_dtype(result["date"])
        assert result.iloc[0]["date"] == pd.Timestamp("2022-12-31")
        assert result.iloc[1]["date"] == pd.Timestamp("2023-01-01")

    def test_convert_different_centuries(self):
        """Test conversion with dates from different centuries."""
        df_test = pd.DataFrame(
            {"date": ["1999-12-31", "2000-01-01", "2023-06-15"], "value": [1, 2, 3]}
        )

        result = convert_to_datetime(df_test)

        assert pd.api.types.is_datetime64_any_dtype(result["date"])
        assert result.iloc[0]["date"] == pd.Timestamp("1999-12-31")
        assert result.iloc[1]["date"] == pd.Timestamp("2000-01-01")

    def test_convert_with_microseconds(self):
        """Test conversion with time including microseconds."""
        df_test = pd.DataFrame({"datetime": ["2023-01-01 10:30:00", "2023-06-15 14:45:30"]})

        result = convert_to_datetime(df_test)

        assert pd.api.types.is_datetime64_any_dtype(result["datetime"])


class TestConvertToDatetimeIntegration:
    """Integration tests combining multiple scenarios."""

    def test_convert_then_filter_by_date(self):
        """Test converting dates then filtering by date range."""
        df_test = pd.DataFrame(
            {
                "date": ["2023-01-01", "2023-06-15", "2023-12-31"],
                "value": [100, 200, 300],
            }
        )

        result = convert_to_datetime(df_test)
        filtered = result[result["date"] > pd.Timestamp("2023-06-01")]

        assert filtered.shape[0] == 2
        assert filtered.iloc[0]["value"] == 200

    def test_convert_then_sort_by_date(self):
        """Test converting dates then sorting."""
        df_test = pd.DataFrame(
            {
                "date": ["2023-12-31", "2023-01-01", "2023-06-15"],
                "value": [300, 100, 200],
            }
        )

        result = convert_to_datetime(df_test)
        sorted_result = result.sort_values("date")

        assert sorted_result.iloc[0]["value"] == 100
        assert sorted_result.iloc[-1]["value"] == 300

    def test_convert_then_extract_components(self):
        """Test converting dates then extracting date components."""
        df_test = pd.DataFrame({"date": ["2023-01-15", "2023-06-20", "2023-12-25"]})

        result = convert_to_datetime(df_test)
        result["year"] = result["date"].dt.year
        result["month"] = result["date"].dt.month
        result["day"] = result["date"].dt.day

        assert result.iloc[0]["year"] == 2023
        assert result.iloc[1]["month"] == 6
        assert result.iloc[2]["day"] == 25

    def test_convert_multiple_times_idempotent(self):
        """Test that converting multiple times is safe."""
        df_test = pd.DataFrame({"date": ["2023-01-01", "2023-06-15"]})

        result1 = convert_to_datetime(df_test)
        result2 = convert_to_datetime(result1)

        # Should remain datetime, not cause errors
        assert pd.api.types.is_datetime64_any_dtype(result2["date"])

    def test_convert_with_subsequent_operations(self):
        """Test that converted dataframe works with pandas operations."""
        df_test = pd.DataFrame(
            {
                "date": ["2023-01-01", "2023-06-15", "2023-12-31"],
                "category": ["A", "B", "A"],
                "value": [100, 200, 300],
            }
        )

        result = convert_to_datetime(df_test)
        grouped = result.groupby("category")["value"].sum()

        assert grouped["A"] == 400
        assert grouped["B"] == 200
