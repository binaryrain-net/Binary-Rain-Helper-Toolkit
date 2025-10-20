"""
Comprehensive test suite for the format_datetime_columns function.
Tests all scenarios, options, edge cases, and error handling.
"""

import pytest
import pandas as pd
from binaryrain_helper_data_processing.dataframe import format_datetime_columns


class TestFormatDatetimeColumnsBasic:
    """Test cases for basic datetime formatting scenarios."""

    def test_format_single_column_default_output(self):
        """Test formatting a single datetime column with default output column."""
        df_test = pd.DataFrame({"date": pd.to_datetime(["2023-01-15", "2023-02-20", "2023-03-25"])})

        result = format_datetime_columns(df_test, ["date"], "%Y-%m-%d")

        assert isinstance(result, pd.DataFrame)
        assert result["date"].iloc[0] == "2023-01-15"
        assert result["date"].iloc[1] == "2023-02-20"
        assert result["date"].iloc[2] == "2023-03-25"

    def test_format_single_column_custom_output(self):
        """Test formatting a single datetime column with custom output column name."""
        df_test = pd.DataFrame({"date": pd.to_datetime(["2023-01-15", "2023-02-20", "2023-03-25"])})

        result = format_datetime_columns(df_test, ["date"], "%Y-%m-%d", ["date_str"])

        assert "date_str" in result.columns
        assert result["date_str"].iloc[0] == "2023-01-15"
        assert result["date_str"].iloc[1] == "2023-02-20"

    def test_format_multiple_columns_default_output(self):
        """Test formatting multiple datetime columns with default output columns."""
        df_test = pd.DataFrame(
            {
                "start_date": pd.to_datetime(["2023-01-15", "2023-02-20"]),
                "end_date": pd.to_datetime(["2023-03-25", "2023-04-30"]),
            }
        )

        result = format_datetime_columns(df_test, ["start_date", "end_date"], "%d/%m/%Y")

        assert result["start_date"].iloc[0] == "15/01/2023"
        assert result["end_date"].iloc[1] == "30/04/2023"

    def test_format_multiple_columns_custom_output(self):
        """Test formatting multiple datetime columns with custom output column names."""
        df_test = pd.DataFrame(
            {
                "start_date": pd.to_datetime(["2023-01-15", "2023-02-20"]),
                "end_date": pd.to_datetime(["2023-03-25", "2023-04-30"]),
            }
        )

        result = format_datetime_columns(
            df_test,
            ["start_date", "end_date"],
            "%d/%m/%Y",
            ["start_str", "end_str"],
        )

        assert "start_str" in result.columns
        assert "end_str" in result.columns
        assert result["start_str"].iloc[0] == "15/01/2023"
        assert result["end_str"].iloc[1] == "30/04/2023"


class TestFormatDatetimeColumnsFormats:
    """Test cases for different datetime format strings."""

    def test_format_ymd_dash(self):
        """Test formatting with YYYY-MM-DD format."""
        df_test = pd.DataFrame({"date": pd.to_datetime(["2023-01-15", "2023-12-31"])})

        result = format_datetime_columns(df_test, ["date"], "%Y-%m-%d")

        assert result["date"].iloc[0] == "2023-01-15"
        assert result["date"].iloc[1] == "2023-12-31"

    def test_format_dmy_slash(self):
        """Test formatting with DD/MM/YYYY format."""
        df_test = pd.DataFrame({"date": pd.to_datetime(["2023-01-15", "2023-12-31"])})

        result = format_datetime_columns(df_test, ["date"], "%d/%m/%Y")

        assert result["date"].iloc[0] == "15/01/2023"
        assert result["date"].iloc[1] == "31/12/2023"

    def test_format_mdy_slash(self):
        """Test formatting with MM/DD/YYYY format."""
        df_test = pd.DataFrame({"date": pd.to_datetime(["2023-01-15", "2023-12-31"])})

        result = format_datetime_columns(df_test, ["date"], "%m/%d/%Y")

        assert result["date"].iloc[0] == "01/15/2023"
        assert result["date"].iloc[1] == "12/31/2023"

    def test_format_with_time(self):
        """Test formatting with datetime including time."""
        df_test = pd.DataFrame(
            {"datetime": pd.to_datetime(["2023-01-15 14:30:45", "2023-12-31 23:59:59"])}
        )

        result = format_datetime_columns(df_test, ["datetime"], "%Y-%m-%d %H:%M:%S")

        assert result["datetime"].iloc[0] == "2023-01-15 14:30:45"
        assert result["datetime"].iloc[1] == "2023-12-31 23:59:59"

    def test_format_iso_format(self):
        """Test formatting with ISO 8601 format."""
        df_test = pd.DataFrame(
            {"datetime": pd.to_datetime(["2023-01-15 14:30:45", "2023-12-31 23:59:59"])}
        )

        result = format_datetime_columns(df_test, ["datetime"], "%Y-%m-%dT%H:%M:%S")

        assert result["datetime"].iloc[0] == "2023-01-15T14:30:45"
        assert result["datetime"].iloc[1] == "2023-12-31T23:59:59"

    def test_format_month_name(self):
        """Test formatting with full month name."""
        df_test = pd.DataFrame({"date": pd.to_datetime(["2023-01-15", "2023-12-31"])})

        result = format_datetime_columns(df_test, ["date"], "%B %d, %Y")

        assert result["date"].iloc[0] == "January 15, 2023"
        assert result["date"].iloc[1] == "December 31, 2023"

    def test_format_abbreviated_month(self):
        """Test formatting with abbreviated month name."""
        df_test = pd.DataFrame({"date": pd.to_datetime(["2023-01-15", "2023-12-31"])})

        result = format_datetime_columns(df_test, ["date"], "%b %d, %Y")

        assert result["date"].iloc[0] == "Jan 15, 2023"
        assert result["date"].iloc[1] == "Dec 31, 2023"

    def test_format_weekday(self):
        """Test formatting with weekday name."""
        df_test = pd.DataFrame({"date": pd.to_datetime(["2023-01-15", "2023-12-31"])})

        result = format_datetime_columns(df_test, ["date"], "%A, %B %d, %Y")

        assert "Sunday" in result["date"].iloc[0]
        assert "Sunday" in result["date"].iloc[1]

    def test_format_year_only(self):
        """Test formatting with year only."""
        df_test = pd.DataFrame({"date": pd.to_datetime(["2023-01-15", "2024-12-31"])})

        result = format_datetime_columns(df_test, ["date"], "%Y")

        assert result["date"].iloc[0] == "2023"
        assert result["date"].iloc[1] == "2024"

    def test_format_time_only(self):
        """Test formatting with time only."""
        df_test = pd.DataFrame(
            {"datetime": pd.to_datetime(["2023-01-15 14:30:45", "2023-12-31 23:59:59"])}
        )

        result = format_datetime_columns(df_test, ["datetime"], "%H:%M:%S")

        assert result["datetime"].iloc[0] == "14:30:45"
        assert result["datetime"].iloc[1] == "23:59:59"


class TestFormatDatetimeColumnsEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_format_single_row(self):
        """Test formatting with single row dataframe."""
        df_test = pd.DataFrame({"date": pd.to_datetime(["2023-01-15"])})

        result = format_datetime_columns(df_test, ["date"], "%Y-%m-%d")

        assert result.shape[0] == 1
        assert result["date"].iloc[0] == "2023-01-15"

    def test_format_large_dataset(self):
        """Test formatting with large dataset."""
        dates = pd.date_range(start="2020-01-01", periods=10000, freq="D")
        df_test = pd.DataFrame({"date": dates})

        result = format_datetime_columns(df_test, ["date"], "%Y-%m-%d")

        assert result.shape[0] == 10000
        assert result["date"].iloc[0] == "2020-01-01"
        assert result["date"].iloc[-1] == "2047-05-18"

    def test_format_preserves_other_columns(self):
        """Test that formatting preserves non-datetime columns."""
        df_test = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "date": pd.to_datetime(["2023-01-15", "2023-02-20", "2023-03-25"]),
                "name": ["Alice", "Bob", "Charlie"],
            }
        )

        result = format_datetime_columns(df_test, ["date"], "%Y-%m-%d")

        assert "id" in result.columns
        assert "name" in result.columns
        assert result["id"].iloc[0] == 1
        assert result["name"].iloc[2] == "Charlie"

    def test_format_with_nat_values(self):
        """Test formatting with NaT (Not a Time) values."""
        df_test = pd.DataFrame({"date": pd.to_datetime(["2023-01-15", None, "2023-03-25"])})

        result = format_datetime_columns(df_test, ["date"], "%Y-%m-%d")

        assert result["date"].iloc[0] == "2023-01-15"
        assert pd.isna(result["date"].iloc[1]) or result["date"].iloc[1] == "NaT"
        assert result["date"].iloc[2] == "2023-03-25"

    def test_format_leap_year_dates(self):
        """Test formatting with leap year dates."""
        df_test = pd.DataFrame({"date": pd.to_datetime(["2020-02-29", "2024-02-29"])})

        result = format_datetime_columns(df_test, ["date"], "%Y-%m-%d")

        assert result["date"].iloc[0] == "2020-02-29"
        assert result["date"].iloc[1] == "2024-02-29"

    def test_format_different_centuries(self):
        """Test formatting with dates from different centuries."""
        df_test = pd.DataFrame({"date": pd.to_datetime(["1999-12-31", "2000-01-01", "2100-01-01"])})

        result = format_datetime_columns(df_test, ["date"], "%Y-%m-%d")

        assert result["date"].iloc[0] == "1999-12-31"
        assert result["date"].iloc[1] == "2000-01-01"
        assert result["date"].iloc[2] == "2100-01-01"

    def test_format_datetime_with_microseconds(self):
        """Test formatting datetime with microseconds."""
        df_test = pd.DataFrame({"datetime": pd.to_datetime(["2023-01-15 14:30:45.123456"])})

        result = format_datetime_columns(df_test, ["datetime"], "%Y-%m-%d %H:%M:%S.%f")

        assert "123456" in result["datetime"].iloc[0]


class TestFormatDatetimeColumnsErrorHandling:
    """Test cases for error handling."""

    def test_mismatched_column_lengths(self):
        """Test error when datetime_columns and datetime_string_columns have different lengths."""
        df_test = pd.DataFrame({"date": pd.to_datetime(["2023-01-15", "2023-02-20"])})

        with pytest.raises(
            ValueError,
            match="The number of datetime columns and datetime string columns must be equal.",
        ):
            format_datetime_columns(df_test, ["date", "another_date"], "%Y-%m-%d", ["date_str"])

    def test_mismatched_more_output_columns(self):
        """Test error when more output columns than input columns."""
        df_test = pd.DataFrame({"date": pd.to_datetime(["2023-01-15", "2023-02-20"])})

        with pytest.raises(
            ValueError,
            match="The number of datetime columns and datetime string columns must be equal.",
        ):
            format_datetime_columns(df_test, ["date"], "%Y-%m-%d", ["date_str", "extra_str"])

    def test_nonexistent_column(self):
        """Test error when trying to format a column that doesn't exist."""
        df_test = pd.DataFrame({"date": pd.to_datetime(["2023-01-15", "2023-02-20"])})

        with pytest.raises(ValueError, match="Error formatting column nonexistent_date:"):
            format_datetime_columns(df_test, ["nonexistent_date"], "%Y-%m-%d")

    def test_non_datetime_column(self):
        """Test error when trying to format a non-datetime column."""
        df_test = pd.DataFrame({"name": ["Bob", "Alice"]})

        with pytest.raises(ValueError, match="Error formatting column name:"):
            format_datetime_columns(df_test, ["name"], "%Y-%m-%d")

    def test_invalid_format_string(self):
        """Test with invalid format string (may give unexpected results)."""
        df_test = pd.DataFrame({"date": pd.to_datetime(["2023-01-15", "2023-02-20"])})

        result = format_datetime_columns(df_test, ["date"], "invalid%%format")

        assert isinstance(result, pd.DataFrame)


class TestFormatDatetimeColumnsMultipleColumns:
    """Test cases for formatting multiple datetime columns."""

    def test_format_three_columns_same_format(self):
        """Test formatting three columns with same format."""
        df_test = pd.DataFrame(
            {
                "created": pd.to_datetime(["2023-01-15", "2023-02-20"]),
                "modified": pd.to_datetime(["2023-01-20", "2023-02-25"]),
                "deleted": pd.to_datetime(["2023-01-25", "2023-03-01"]),
            }
        )

        result = format_datetime_columns(df_test, ["created", "modified", "deleted"], "%Y-%m-%d")

        assert result["created"].iloc[0] == "2023-01-15"
        assert result["modified"].iloc[0] == "2023-01-20"
        assert result["deleted"].iloc[0] == "2023-01-25"

    def test_format_multiple_columns_different_output_names(self):
        """Test formatting multiple columns with different output column names."""
        df_test = pd.DataFrame(
            {
                "created": pd.to_datetime(["2023-01-15", "2023-02-20"]),
                "modified": pd.to_datetime(["2023-01-20", "2023-02-25"]),
            }
        )

        result = format_datetime_columns(
            df_test,
            ["created", "modified"],
            "%d/%m/%Y",
            ["created_str", "modified_str"],
        )

        assert "created_str" in result.columns
        assert "modified_str" in result.columns
        assert result["created_str"].iloc[0] == "15/01/2023"
        assert result["modified_str"].iloc[1] == "25/02/2023"

    def test_format_subset_of_datetime_columns(self):
        """Test formatting only a subset of datetime columns in dataframe."""
        df_test = pd.DataFrame(
            {
                "created": pd.to_datetime(["2023-01-15", "2023-02-20"]),
                "modified": pd.to_datetime(["2023-01-20", "2023-02-25"]),
                "deleted": pd.to_datetime(["2023-01-25", "2023-03-01"]),
            }
        )

        result = format_datetime_columns(df_test, ["created", "modified"], "%Y-%m-%d")

        assert result["created"].iloc[0] == "2023-01-15"
        assert result["modified"].iloc[0] == "2023-01-20"
        assert isinstance(result["deleted"].iloc[0], pd.Timestamp)


class TestFormatDatetimeColumnsDataPreservation:
    """Test cases for data preservation during formatting."""

    def test_original_dataframe_not_modified(self):
        """Test that original dataframe is modified in place (pandas behavior)."""
        df_test = pd.DataFrame({"date": pd.to_datetime(["2023-01-15", "2023-02-20"])})

        result = format_datetime_columns(df_test, ["date"], "%Y-%m-%d")

        assert isinstance(result["date"].iloc[0], str)
        assert isinstance(df_test["date"].iloc[0], str)

    def test_format_creates_new_column(self):
        """Test that formatting with new column name preserves original."""
        df_test = pd.DataFrame({"date": pd.to_datetime(["2023-01-15", "2023-02-20"])})

        result = format_datetime_columns(df_test, ["date"], "%Y-%m-%d", ["date_str"])

        assert "date" in result.columns
        assert "date_str" in result.columns
        assert isinstance(result["date"].iloc[0], pd.Timestamp)
        assert isinstance(result["date_str"].iloc[0], str)

    def test_format_overwrites_existing_column(self):
        """Test that formatting can overwrite an existing column."""
        df_test = pd.DataFrame(
            {
                "date": pd.to_datetime(["2023-01-15", "2023-02-20"]),
                "date_str": ["old_value", "old_value"],
            }
        )

        result = format_datetime_columns(df_test, ["date"], "%Y-%m-%d", ["date_str"])

        assert result["date_str"].iloc[0] == "2023-01-15"
        assert result["date_str"].iloc[0] != "old_value"

    def test_index_preservation(self):
        """Test that dataframe index is preserved during formatting."""
        df_test = pd.DataFrame(
            {"date": pd.to_datetime(["2023-01-15", "2023-02-20", "2023-03-25"])},
            index=[10, 20, 30],
        )

        result = format_datetime_columns(df_test, ["date"], "%Y-%m-%d")

        assert list(result.index) == [10, 20, 30]

    def test_column_order_preservation(self):
        """Test that column order is preserved during formatting."""
        df_test = pd.DataFrame(
            {
                "id": [1, 2],
                "date": pd.to_datetime(["2023-01-15", "2023-02-20"]),
                "name": ["Alice", "Bob"],
            }
        )

        result = format_datetime_columns(df_test, ["date"], "%Y-%m-%d")

        assert list(result.columns) == ["id", "date", "name"]


class TestFormatDatetimeColumnsSpecialFormats:
    """Test cases for special and complex format strings."""

    def test_format_with_literal_text(self):
        """Test formatting with literal text in format string."""
        df_test = pd.DataFrame({"date": pd.to_datetime(["2023-01-15", "2023-02-20"])})

        result = format_datetime_columns(df_test, ["date"], "Date: %Y-%m-%d")

        assert result["date"].iloc[0] == "Date: 2023-01-15"
        assert result["date"].iloc[1] == "Date: 2023-02-20"

    def test_format_12_hour_time(self):
        """Test formatting with 12-hour time format."""
        df_test = pd.DataFrame(
            {"datetime": pd.to_datetime(["2023-01-15 14:30:45", "2023-01-15 08:15:30"])}
        )

        result = format_datetime_columns(df_test, ["datetime"], "%I:%M %p")

        assert result["datetime"].iloc[0] == "02:30 PM"
        assert result["datetime"].iloc[1] == "08:15 AM"

    def test_format_day_of_year(self):
        """Test formatting with day of year."""
        df_test = pd.DataFrame({"date": pd.to_datetime(["2023-01-01", "2023-12-31"])})

        result = format_datetime_columns(df_test, ["date"], "%j")

        assert result["date"].iloc[0] == "001"
        assert result["date"].iloc[1] == "365"

    def test_format_week_number(self):
        """Test formatting with week number."""
        df_test = pd.DataFrame({"date": pd.to_datetime(["2023-01-01", "2023-12-31"])})

        result = format_datetime_columns(df_test, ["date"], "Week %U of %Y")

        assert "Week" in result["date"].iloc[0]
        assert "2023" in result["date"].iloc[0]


class TestFormatDatetimeColumnsIntegration:
    """Integration tests combining multiple scenarios."""

    def test_format_mixed_dataframe(self):
        """Test formatting in a dataframe with mixed column types."""
        df_test = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "name": ["Alice", "Bob", "Charlie"],
                "created": pd.to_datetime(["2023-01-15", "2023-02-20", "2023-03-25"]),
                "value": [100.5, 200.75, 300],
                "active": [True, False, True],
            }
        )

        result = format_datetime_columns(df_test, ["created"], "%Y-%m-%d")

        assert result.shape == (3, 5)
        assert result["created"].iloc[0] == "2023-01-15"
        assert result["id"].iloc[0] == 1
        assert result["name"].iloc[1] == "Bob"
        assert result["value"].iloc[2] == 300

    def test_format_multiple_times_same_column(self):
        """Test formatting the same column multiple times with different formats."""
        df_test = pd.DataFrame({"date": pd.to_datetime(["2023-01-15", "2023-02-20"])})

        result = format_datetime_columns(df_test, ["date"], "%Y-%m-%d", ["date_ymd"])
        result = format_datetime_columns(result, ["date"], "%d/%m/%Y", ["date_dmy"])

        assert "date_ymd" in result.columns
        assert "date_dmy" in result.columns
        assert result["date_ymd"].iloc[0] == "2023-01-15"
        assert result["date_dmy"].iloc[0] == "15/01/2023"

    def test_complex_workflow(self):
        """Test complex workflow with multiple datetime columns and formats."""
        df_test = pd.DataFrame(
            {
                "user_id": [1, 2, 3],
                "registration_date": pd.to_datetime(["2023-01-15", "2023-02-20", "2023-03-25"]),
                "last_login": pd.to_datetime(
                    ["2023-06-15 14:30:00", "2023-06-20 09:15:00", "2023-06-25 18:45:00"]
                ),
                "subscription_end": pd.to_datetime(["2024-01-15", "2024-02-20", "2024-03-25"]),
            }
        )

        result = format_datetime_columns(
            df_test,
            ["registration_date", "subscription_end"],
            "%B %d, %Y",
            ["reg_date_str", "sub_end_str"],
        )
        result = format_datetime_columns(
            result, ["last_login"], "%Y-%m-%d %H:%M", ["last_login_str"]
        )

        assert "reg_date_str" in result.columns
        assert "sub_end_str" in result.columns
        assert "last_login_str" in result.columns
        assert result["reg_date_str"].iloc[0] == "January 15, 2023"
        assert result["last_login_str"].iloc[1] == "2023-06-20 09:15"
