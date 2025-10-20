"""
Comprehensive test suite for the format_numeric_to_string function.
Tests all scenarios, separator handling, edge cases, and error handling.
"""

import pytest
import pandas as pd
from binaryrain_helper_data_processing.dataframe import format_numeric_to_string


class TestFormatNumericToStringBasic:
    """Test cases for basic numeric to string formatting."""

    def test_format_single_column_default_params(self):
        """Test formatting a single column with default parameters."""
        df_test = pd.DataFrame({"value": [1000.50, 2000.75, 3000.25], "label": ["A", "B", "C"]})

        result = format_numeric_to_string(df_test, ["value"])

        assert isinstance(result, pd.DataFrame)
        assert result["value"].dtype == "object"
        assert result["value"].iloc[0] == "1.000,50"
        assert result["value"].iloc[1] == "2.000,75"
        assert result["value"].iloc[2] == "3.000,25"

    def test_format_multiple_columns(self):
        """Test formatting multiple columns."""
        df_test = pd.DataFrame({"val1": [100.5, 200.75], "val2": [1.5, 2.5]})

        result = format_numeric_to_string(df_test, ["val1", "val2"])

        assert result["val1"].iloc[0] == "100,50"
        assert result["val2"].iloc[0] == "1,50"

    def test_format_integers(self):
        """Test formatting integer values."""
        df_test = pd.DataFrame({"value": [1000, 2000, 3000]})

        result = format_numeric_to_string(df_test, ["value"])

        assert result["value"].iloc[0] == "1.000,00"
        assert result["value"].iloc[1] == "2.000,00"
        assert result["value"].iloc[2] == "3.000,00"

    def test_format_floats(self):
        """Test formatting float values."""
        df_test = pd.DataFrame({"value": [1234.567, 5678.901]})

        result = format_numeric_to_string(df_test, ["value"])

        assert result["value"].iloc[0] == "1.234,57"
        assert result["value"].iloc[1] == "5.678,90"

    def test_preserves_non_target_columns(self):
        """Test that non-target columns are preserved unchanged."""
        df_test = pd.DataFrame(
            {"id": [1, 2, 3], "value": [1000.50, 2000.75, 3000.25], "name": ["A", "B", "C"]}
        )

        result = format_numeric_to_string(df_test, ["value"])

        assert result["id"].iloc[0] == 1
        assert result["name"].iloc[0] == "A"


class TestFormatNumericToStringDecimalPlaces:
    """Test cases for decimal places parameter."""

    def test_zero_decimal_places(self):
        """Test formatting with zero decimal places."""
        df_test = pd.DataFrame({"value": [1000.567, 2000.89]})

        result = format_numeric_to_string(df_test, ["value"], decimal_places=0)

        assert result["value"].iloc[0] == "1.001"
        assert result["value"].iloc[1] == "2.001"

    def test_one_decimal_place(self):
        """Test formatting with one decimal place."""
        df_test = pd.DataFrame({"value": [1000.567, 2000.89]})

        result = format_numeric_to_string(df_test, ["value"], decimal_places=1)

        assert result["value"].iloc[0] == "1.000,6"
        assert result["value"].iloc[1] == "2.000,9"

    def test_three_decimal_places(self):
        """Test formatting with three decimal places."""
        df_test = pd.DataFrame({"value": [1000.5678, 2000.1234]})

        result = format_numeric_to_string(df_test, ["value"], decimal_places=3)

        assert result["value"].iloc[0] == "1.000,568"
        assert result["value"].iloc[1] == "2.000,123"

    def test_five_decimal_places(self):
        """Test formatting with five decimal places."""
        df_test = pd.DataFrame({"value": [1000.123456]})

        result = format_numeric_to_string(df_test, ["value"], decimal_places=5)

        assert result["value"].iloc[0] == "1.000,12346"

    def test_rounding_behavior(self):
        """Test that rounding works correctly."""
        df_test = pd.DataFrame({"value": [1.555, 1.545, 1.5]})

        result = format_numeric_to_string(df_test, ["value"], decimal_places=2)

        assert result["value"].iloc[0] == "1,56"
        assert result["value"].iloc[1] == "1,55"
        assert result["value"].iloc[2] == "1,50"


class TestFormatNumericToStringCustomSeparators:
    """Test cases for custom separator configurations."""

    def test_us_format(self):
        """Test US format (comma thousands, dot decimal)."""
        df_test = pd.DataFrame({"value": [1234.56, 5678.90]})

        result = format_numeric_to_string(
            df_test, ["value"], decimal_separator=".", thousands_separator=","
        )

        assert result["value"].iloc[0] == "1,234.56"
        assert result["value"].iloc[1] == "5,678.90"

    def test_space_as_thousands_separator(self):
        """Test with space as thousands separator."""
        df_test = pd.DataFrame({"value": [1234.56, 5678.90]})

        result = format_numeric_to_string(
            df_test, ["value"], decimal_separator=",", thousands_separator=" "
        )

        assert result["value"].iloc[0] == "1 234,56"
        assert result["value"].iloc[1] == "5 678,90"

    def test_custom_temp_separator(self):
        """Test with custom temporary separator."""
        df_test = pd.DataFrame({"value": [1234.56, 5678.90]})

        result = format_numeric_to_string(df_test, ["value"], temp_separator="#")

        assert result["value"].iloc[0] == "1.234,56"
        assert result["value"].iloc[1] == "5.678,90"

    def test_apostrophe_separator(self):
        """Test with apostrophe as thousands separator (Swiss format)."""
        df_test = pd.DataFrame({"value": [1234.56, 5678.90]})

        result = format_numeric_to_string(
            df_test, ["value"], decimal_separator=".", thousands_separator="'"
        )

        assert result["value"].iloc[0] == "1'234.56"
        assert result["value"].iloc[1] == "5'678.90"


class TestFormatNumericToStringParsingOldFormats:
    """Test cases for parsing old format strings."""

    def test_parse_us_format_to_european(self):
        """Test parsing US format strings to European format."""
        df_test = pd.DataFrame({"value": ["1,234.56", "5,678.90"]})

        result = format_numeric_to_string(
            df_test,
            ["value"],
            decimal_separator=",",
            thousands_separator=".",
            old_decimal_separator=".",
            old_thousands_separator=",",
        )

        assert result["value"].iloc[0] == "1.234,56"
        assert result["value"].iloc[1] == "5.678,90"

    def test_parse_european_format_to_us(self):
        """Test parsing European format strings to US format."""
        df_test = pd.DataFrame({"value": ["1.234,56", "5.678,90"]})

        result = format_numeric_to_string(
            df_test,
            ["value"],
            decimal_separator=".",
            thousands_separator=",",
            old_decimal_separator=",",
            old_thousands_separator=".",
        )

        assert result["value"].iloc[0] == "1,234.56"
        assert result["value"].iloc[1] == "5,678.90"

    def test_parse_space_separated_format(self):
        """Test parsing space-separated format."""
        df_test = pd.DataFrame({"value": ["1 234,56", "5 678,90"]})

        result = format_numeric_to_string(
            df_test,
            ["value"],
            decimal_separator=",",
            thousands_separator=".",
            old_decimal_separator=",",
            old_thousands_separator=" ",
        )

        assert result["value"].iloc[0] == "1.234,56"
        assert result["value"].iloc[1] == "5.678,90"

    def test_parse_no_thousands_separator(self):
        """Test parsing numbers without thousands separator."""
        df_test = pd.DataFrame({"value": ["1234.56", "5678.90"]})

        result = format_numeric_to_string(
            df_test,
            ["value"],
            decimal_separator=",",
            thousands_separator=".",
            old_decimal_separator=".",
            old_thousands_separator="",
        )

        assert result["value"].iloc[0] == "1.234,56"
        assert result["value"].iloc[1] == "5.678,90"


class TestFormatNumericToStringEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_dataframe(self):
        """Test with empty dataframe."""
        df_test = pd.DataFrame({"value": []})

        result = format_numeric_to_string(df_test, ["value"])
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_single_row(self):
        """Test with single row dataframe."""
        df_test = pd.DataFrame({"value": [1234.56]})

        result = format_numeric_to_string(df_test, ["value"])

        assert result.shape[0] == 1
        assert result["value"].iloc[0] == "1.234,56"

    def test_zero_values(self):
        """Test with zero values."""
        df_test = pd.DataFrame({"value": [0, 0.0, 0.00]})

        result = format_numeric_to_string(df_test, ["value"])

        assert result["value"].iloc[0] == "0,00"
        assert result["value"].iloc[1] == "0,00"
        assert result["value"].iloc[2] == "0,00"

    def test_negative_numbers(self):
        """Test with negative numbers."""
        df_test = pd.DataFrame({"value": [-1234.56, -5678.90]})

        result = format_numeric_to_string(df_test, ["value"])

        assert result["value"].iloc[0] == "-1.234,56"
        assert result["value"].iloc[1] == "-5.678,90"

    def test_very_large_numbers(self):
        """Test with very large numbers."""
        df_test = pd.DataFrame({"value": [1234567890.12, 9876543210.99]})

        result = format_numeric_to_string(df_test, ["value"])

        assert result["value"].iloc[0] == "1.234.567.890,12"
        assert result["value"].iloc[1] == "9.876.543.210,99"

    def test_very_small_numbers(self):
        """Test with very small numbers."""
        df_test = pd.DataFrame({"value": [0.001, 0.0001]})

        result = format_numeric_to_string(df_test, ["value"])

        assert result["value"].iloc[0] == "0,00"
        assert result["value"].iloc[1] == "0,00"

    def test_very_small_numbers_high_precision(self):
        """Test very small numbers with high decimal precision."""
        df_test = pd.DataFrame({"value": [0.001, 0.0001]})

        result = format_numeric_to_string(df_test, ["value"], decimal_places=4)

        assert result["value"].iloc[0] == "0,0010"
        assert result["value"].iloc[1] == "0,0001"


class TestFormatNumericToStringInvalidData:
    """Test cases for invalid data handling."""

    def test_nan_values(self):
        """Test that NaN values become empty strings."""
        df_test = pd.DataFrame({"value": [1234.56, float("nan"), 5678.90]})

        result = format_numeric_to_string(df_test, ["value"])

        assert result["value"].iloc[0] == "1.234,56"
        assert result["value"].iloc[1] == ""
        assert result["value"].iloc[2] == "5.678,90"

    def test_none_values(self):
        """Test that None values become empty strings."""
        df_test = pd.DataFrame({"value": [1234.56, None, 5678.90]})

        result = format_numeric_to_string(df_test, ["value"])

        assert result["value"].iloc[0] == "1.234,56"
        assert result["value"].iloc[1] == ""
        assert result["value"].iloc[2] == "5.678,90"

    def test_empty_strings(self):
        """Test that empty strings become empty strings."""
        df_test = pd.DataFrame({"value": ["1234.56", "", "5678.90"]})

        result = format_numeric_to_string(
            df_test,
            ["value"],
            decimal_separator=",",
            thousands_separator=".",
            old_decimal_separator=".",
            old_thousands_separator=",",
        )

        assert result["value"].iloc[0] == "1.234,56"
        assert result["value"].iloc[1] == ""
        assert result["value"].iloc[2] == "5.678,90"

    def test_invalid_strings(self):
        """Test that invalid strings become empty strings."""
        df_test = pd.DataFrame({"value": ["abc", "def", "1234.56"]})

        result = format_numeric_to_string(
            df_test,
            ["value"],
            decimal_separator=",",
            thousands_separator=".",
            old_decimal_separator=".",
            old_thousands_separator=",",
        )

        assert result["value"].iloc[0] == ""
        assert result["value"].iloc[1] == ""
        assert result["value"].iloc[2] == "1.234,56"

    def test_mixed_valid_invalid(self):
        """Test with mixed valid and invalid values."""
        df_test = pd.DataFrame({"value": [1234.56, "invalid", 5678.90, None]})

        result = format_numeric_to_string(df_test, ["value"])

        assert result["value"].iloc[0] == "1.234,56"
        assert result["value"].iloc[1] == ""
        assert result["value"].iloc[2] == "5.678,90"
        assert result["value"].iloc[3] == ""


class TestFormatNumericToStringErrorHandling:
    """Test cases for error handling."""

    def test_empty_column_list(self):
        """Test with empty column list returns dataframe unchanged."""
        df_test = pd.DataFrame({"value": [1234.56, 5678.91]})

        result = format_numeric_to_string(df_test, [])

        assert str(result["value"].iloc[0]) == "1234.56"
        assert str(result["value"].iloc[1]) == "5678.91"

    def test_nonexistent_column(self):
        """Test error when column doesn't exist."""
        df_test = pd.DataFrame({"value": [1234.56]})

        with pytest.raises(KeyError, match="Columns not found in dataframe"):
            format_numeric_to_string(df_test, ["nonexistent"])

    def test_multiple_nonexistent_columns(self):
        """Test error with multiple nonexistent columns."""
        df_test = pd.DataFrame({"value": [1234.56]})

        with pytest.raises(KeyError, match="Columns not found in dataframe"):
            format_numeric_to_string(df_test, ["col1", "col2"])

    def test_negative_decimal_places(self):
        """Test error with negative decimal places."""
        df_test = pd.DataFrame({"value": [1234.56]})

        with pytest.raises(ValueError, match="decimal_places must be >= 0"):
            format_numeric_to_string(df_test, ["value"], decimal_places=-1)

    def test_same_decimal_and_thousands_separator(self):
        """Test error when decimal and thousands separators are the same."""
        df_test = pd.DataFrame({"value": [1234.56]})

        with pytest.raises(
            ValueError, match="decimal_separator and thousands_separator must differ"
        ):
            format_numeric_to_string(
                df_test, ["value"], decimal_separator=",", thousands_separator=","
            )

    def test_temp_separator_conflicts_with_decimal(self):
        """Test error when temp separator conflicts with decimal separator."""
        df_test = pd.DataFrame({"value": [1234.56]})

        with pytest.raises(
            ValueError,
            match="temp_separator must differ from decimal and thousands separators",
        ):
            format_numeric_to_string(df_test, ["value"], decimal_separator=",", temp_separator=",")

    def test_temp_separator_conflicts_with_thousands(self):
        """Test error when temp separator conflicts with thousands separator."""
        df_test = pd.DataFrame({"value": [1234.56]})

        with pytest.raises(
            ValueError,
            match="temp_separator must differ from decimal and thousands separators",
        ):
            format_numeric_to_string(
                df_test, ["value"], thousands_separator=".", temp_separator="."
            )


class TestFormatNumericToStringDataPreservation:
    """Test cases for data preservation during formatting."""

    def test_preserves_column_order(self):
        """Test that column order is preserved."""
        df_test = pd.DataFrame({"col3": [100.5], "col1": ["A"], "col2": [1000.75]})

        result = format_numeric_to_string(df_test, ["col3", "col2"])

        assert list(result.columns) == ["col3", "col1", "col2"]

    def test_preserves_index(self):
        """Test that dataframe index is preserved."""
        df_test = pd.DataFrame({"value": [100.5, 200.75, 300.25]}, index=[10, 20, 30])

        result = format_numeric_to_string(df_test, ["value"])

        assert list(result.index) == [10, 20, 30]

    def test_mutates_original_dataframe(self):
        """Test that original dataframe is mutated."""
        df_test = pd.DataFrame({"value": [1234.56]})

        result = format_numeric_to_string(df_test, ["value"])

        assert result is df_test
        assert df_test["value"].iloc[0] == "1.234,56"

    def test_preserves_non_formatted_columns(self):
        """Test that columns not in format list are preserved."""
        df_test = pd.DataFrame({"value1": [1234.56], "value2": [5678.91], "label": ["A"]})

        result = format_numeric_to_string(df_test, ["value1"])

        assert result["value1"].iloc[0] == "1.234,56"
        assert str(result["value2"].iloc[0]) == "5678.91"
        assert result["label"].iloc[0] == "A"


class TestFormatNumericToStringSingleColumn:
    """Test cases for single column operations."""

    def test_format_only_specified_column(self):
        """Test that only specified column is formatted."""
        df_test = pd.DataFrame({"value1": [1234.56], "value2": [5678.90]})

        result = format_numeric_to_string(df_test, ["value1"])

        assert result["value1"].dtype == "object"
        assert result["value2"].dtype == "float64"

    def test_single_numeric_column(self):
        """Test single numeric column formatting."""
        df_test = pd.DataFrame({"price": [1234.567, 5678.901]})

        result = format_numeric_to_string(df_test, ["price"])

        assert result["price"].iloc[0] == "1.234,57"
        assert result["price"].iloc[1] == "5.678,90"

    def test_single_string_column(self):
        """Test single string column formatting."""
        df_test = pd.DataFrame({"price": ["1234.56", "5678.90"]})

        result = format_numeric_to_string(
            df_test,
            ["price"],
            decimal_separator=",",
            thousands_separator=".",
            old_decimal_separator=".",
            old_thousands_separator=",",
        )

        assert result["price"].iloc[0] == "1.234,56"
        assert result["price"].iloc[1] == "5.678,90"


class TestFormatNumericToStringMultipleColumns:
    """Test cases for multiple column operations."""

    def test_format_three_columns(self):
        """Test formatting three columns simultaneously."""
        df_test = pd.DataFrame(
            {"val1": [100.5, 200.75], "val2": [1.5, 2.5], "val3": [1000.25, 2000.50]}
        )

        result = format_numeric_to_string(df_test, ["val1", "val2", "val3"])

        assert result["val1"].iloc[0] == "100,50"
        assert result["val2"].iloc[0] == "1,50"
        assert result["val3"].iloc[0] == "1.000,25"

    def test_format_all_numeric_columns(self):
        """Test formatting all numeric columns in dataframe."""
        df_test = pd.DataFrame({"col1": [100.5], "col2": [300.75], "col3": [500.25]})

        result = format_numeric_to_string(df_test, ["col1", "col2", "col3"])

        assert result["col1"].dtype == "object"
        assert result["col2"].dtype == "object"
        assert result["col3"].dtype == "object"

    def test_format_subset_of_columns(self):
        """Test formatting a subset of columns."""
        df_test = pd.DataFrame({"id": [1], "value1": [1000.5], "value2": [3000.75], "name": ["A"]})

        result = format_numeric_to_string(df_test, ["value1", "value2"])

        assert result["value1"].dtype == "object"
        assert result["value2"].dtype == "object"
        assert result["id"].dtype in ["int64", "int32"]
        assert result["name"].dtype == "object"


class TestFormatNumericToStringWhitespace:
    """Test cases for whitespace handling."""

    def test_leading_whitespace(self):
        """Test with leading whitespace in string values."""
        df_test = pd.DataFrame({"value": ["  1234.56", " 5678.90"]})

        result = format_numeric_to_string(
            df_test,
            ["value"],
            decimal_separator=",",
            thousands_separator=".",
            old_decimal_separator=".",
            old_thousands_separator=",",
        )

        assert result["value"].iloc[0] == "1.234,56"
        assert result["value"].iloc[1] == "5.678,90"

    def test_trailing_whitespace(self):
        """Test with trailing whitespace in string values."""
        df_test = pd.DataFrame({"value": ["1234.56  ", "5678.90 "]})

        result = format_numeric_to_string(
            df_test,
            ["value"],
            decimal_separator=",",
            thousands_separator=".",
            old_decimal_separator=".",
            old_thousands_separator=",",
        )

        assert result["value"].iloc[0] == "1.234,56"
        assert result["value"].iloc[1] == "5.678,90"

    def test_mixed_whitespace(self):
        """Test with mixed whitespace in string values."""
        df_test = pd.DataFrame({"value": ["  1234.56  ", " 5678.90 "]})

        result = format_numeric_to_string(
            df_test,
            ["value"],
            decimal_separator=",",
            thousands_separator=".",
            old_decimal_separator=".",
            old_thousands_separator=",",
        )

        assert result["value"].iloc[0] == "1.234,56"
        assert result["value"].iloc[1] == "5.678,90"


class TestFormatNumericToStringIntegration:
    """Integration tests combining multiple scenarios."""

    def test_real_world_financial_data(self):
        """Test with real-world financial data format."""
        df_test = pd.DataFrame(
            {
                "transaction_id": ["T001", "T002", "T003"],
                "amount": [1234.56, 5678.90, 123.45],
                "tax": [234.56, 1078.90, 23.45],
                "description": ["Sale", "Purchase", "Refund"],
            }
        )

        result = format_numeric_to_string(df_test, ["amount", "tax"])

        assert result["amount"].iloc[0] == "1.234,56"
        assert result["tax"].iloc[0] == "234,56"
        assert result["description"].iloc[0] == "Sale"

    def test_parse_and_reformat(self):
        """Test parsing one format and reformatting to another."""
        df_test = pd.DataFrame({"price": ["1,234.56", "5,678.90"]})

        result = format_numeric_to_string(
            df_test,
            ["price"],
            decimal_separator=",",
            thousands_separator=" ",
            old_decimal_separator=".",
            old_thousands_separator=",",
        )

        assert result["price"].iloc[0] == "1 234,56"
        assert result["price"].iloc[1] == "5 678,90"

    def test_mixed_numeric_and_string_input(self):
        """Test with mixed numeric and string input."""
        df_test = pd.DataFrame({"value": [1234.56, "5678.90", 9012.34]})

        result = format_numeric_to_string(
            df_test,
            ["value"],
            decimal_separator=",",
            thousands_separator=".",
            old_decimal_separator=".",
            old_thousands_separator=",",
        )

        assert result["value"].iloc[0] == "1.234,56"
        assert result["value"].iloc[1] == "5.678,90"
        assert result["value"].iloc[2] == "9.012,34"

    def test_large_dataset_performance(self):
        """Test with large dataset."""
        data = {"value": [1000.50 + i for i in range(1000)]}
        df_test = pd.DataFrame(data)

        result = format_numeric_to_string(df_test, ["value"])

        assert result.shape[0] == 1000
        assert result["value"].iloc[0] == "1.000,50"
        assert result["value"].iloc[999] == "1.999,50"

    def test_roundtrip_conversion(self):
        """Test that values can be parsed back after formatting."""
        df_test = pd.DataFrame({"value": [1234.56, 5678.90]})

        result = format_numeric_to_string(df_test, ["value"])

        # Parse back
        parsed = format_numeric_to_string(
            result,
            ["value"],
            decimal_separator=".",
            thousands_separator=",",
            old_decimal_separator=",",
            old_thousands_separator=".",
            decimal_places=2,
        )

        assert parsed["value"].iloc[0] == "1,234.56"
        assert parsed["value"].iloc[1] == "5,678.90"
