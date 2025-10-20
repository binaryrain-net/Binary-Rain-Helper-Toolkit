"""
Comprehensive test suite for the remove_empty_values function.
Tests all scenarios, edge cases, and filtering operations.
"""

import pytest
import pandas as pd
from binaryrain_helper_data_processing.dataframe import remove_empty_values


class TestRemoveEmptyValuesBasic:
    """Test cases for basic empty value removal scenarios."""

    def test_remove_empty_strings(self):
        """Test removing rows with empty strings in filter column."""
        df_test = pd.DataFrame(
            {
                "name": ["Alice", "", "Charlie", "David"],
                "age": [25, 30, 35, 40],
                "city": ["New York", "Los Angeles", "Chicago", "Boston"],
            }
        )

        result = remove_empty_values(df_test, "name")

        assert isinstance(result, pd.DataFrame)
        assert len(result["name"].to_numpy()) == 3
        assert "Alice" in result["name"].to_numpy()
        assert "Charlie" in result["name"].to_numpy()
        assert "David" in result["name"].to_numpy()
        assert "" not in result["name"].to_numpy()

    def test_remove_na_values(self):
        """Test removing rows with NA values in filter column."""
        df_test = pd.DataFrame(
            {
                "name": ["Alice", None, "Charlie", "David"],
                "age": [25, 30, 35, 40],
            }
        )

        result = remove_empty_values(df_test, "name")

        assert result.shape[0] == 3
        assert "Alice" in result["name"].to_numpy()
        assert "Charlie" in result["name"].to_numpy()
        assert "David" in result["name"].to_numpy()

    def test_remove_nan_values(self):
        """Test removing rows with NaN values in filter column."""
        df_test = pd.DataFrame(
            {
                "value": [1, float("nan"), 3, 4],
                "category": ["A", "B", "C", "D"],
            }
        )

        result = remove_empty_values(df_test, "value")

        assert result.shape[0] == 3
        assert result["value"].iloc[0] == 1
        assert result["value"].iloc[1] == 3
        assert result["value"].iloc[2] == 4

    def test_no_empty_values(self):
        """Test when there are no empty values to remove."""
        df_test = pd.DataFrame(
            {
                "name": ["Alice", "Bob", "Charlie"],
                "age": [25, 30, 35],
            }
        )

        result = remove_empty_values(df_test, "name")

        assert result.shape[0] == 3
        assert list(result["name"]) == ["Alice", "Bob", "Charlie"]

    def test_all_empty_values(self):
        """Test when all values in filter column are empty."""
        df_test = pd.DataFrame(
            {
                "name": ["", None, "", None],
                "age": [25, 30, 35, 40],
            }
        )

        result = remove_empty_values(df_test, "name")

        assert isinstance(result, pd.DataFrame)
        assert result.empty
        assert list(result.columns) == ["name", "age"]


class TestRemoveEmptyValuesIndexReset:
    """Test cases for index reset after filtering."""

    def test_index_reset_after_removal(self):
        """Test that index is reset after removing empty values."""
        df_test = pd.DataFrame(
            {
                "name": ["Alice", "", "Charlie", None, "Eve"],
                "age": [25, 30, 35, 40, 45],
            },
            index=[10, 20, 30, 40, 50],
        )

        result = remove_empty_values(df_test, "name")

        assert list(result.index) == [0, 1, 2]
        assert result.iloc[0]["name"] == "Alice"
        assert result.iloc[1]["name"] == "Charlie"
        assert result.iloc[2]["name"] == "Eve"

    def test_index_reset_continuous(self):
        """Test that index is continuous after reset."""
        df_test = pd.DataFrame(
            {
                "value": [1, None, 3, None, 5, None, 7],
                "label": ["A", "B", "C", "D", "E", "F", "G"],
            }
        )

        result = remove_empty_values(df_test, "value")

        assert list(result.index) == [0, 1, 2, 3]

    def test_index_reset_preserves_order(self):
        """Test that row order is preserved after index reset."""
        df_test = pd.DataFrame(
            {
                "name": ["Alice", None, "Bob", "", "Charlie"],
                "order": [1, 2, 3, 4, 5],
            }
        )

        result = remove_empty_values(df_test, "name")

        assert list(result["order"]) == [1, 3, 5]
        assert list(result["name"]) == ["Alice", "Bob", "Charlie"]


class TestRemoveEmptyValuesFilterColumn:
    """Test cases for different filter column scenarios."""

    def test_filter_on_string_column(self):
        """Test filtering on a string column."""
        df_test = pd.DataFrame(
            {
                "name": ["Alice", "", "Charlie"],
                "email": ["alice@test.com", "bob@test.com", "charlie@test.com"],
                "age": [25, 30, 35],
            }
        )

        result = remove_empty_values(df_test, "name")

        assert result.shape[0] == 2
        assert "bob@test.com" not in result["email"].to_numpy()

    def test_filter_on_numeric_column(self):
        """Test filtering on a numeric column."""
        df_test = pd.DataFrame(
            {
                "id": [1, None, 3, 4],
                "name": ["Alice", "Bob", "Charlie", "David"],
            }
        )

        result = remove_empty_values(df_test, "id")

        assert result.shape[0] == 3
        assert "Bob" not in result["name"].to_numpy()

    def test_filter_on_first_column(self):
        """Test filtering on the first column."""
        df_test = pd.DataFrame(
            {
                "id": [1, None, 3],
                "name": ["Alice", "Bob", "Charlie"],
                "age": [25, 30, 35],
            }
        )

        result = remove_empty_values(df_test, "id")

        assert result.shape[0] == 2

    def test_filter_on_last_column(self):
        """Test filtering on the last column."""
        df_test = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "name": ["Alice", "Bob", "Charlie"],
                "status": ["active", "", "active"],
            }
        )

        result = remove_empty_values(df_test, "status")

        assert result.shape[0] == 2
        assert "Bob" not in result["name"].to_numpy()

    def test_filter_on_middle_column(self):
        """Test filtering on a middle column."""
        df_test = pd.DataFrame(
            {
                "id": [1, 2, 3, 4],
                "name": ["Alice", "", "Charlie", "David"],
                "age": [25, 30, 35, 40],
            }
        )

        result = remove_empty_values(df_test, "name")

        assert result.shape[0] == 3


class TestRemoveEmptyValuesEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_dataframe(self):
        """Test with empty dataframe."""
        df_test = pd.DataFrame()

        with pytest.raises(KeyError, match="Column 'name' not found in DataFrame."):
            remove_empty_values(df_test, "name")

    def test_single_row_kept(self):
        """Test with single row that is kept."""
        df_test = pd.DataFrame({"name": ["Alice"], "age": [25]})

        result = remove_empty_values(df_test, "name")

        assert result.shape[0] == 1
        assert result.iloc[0]["name"] == "Alice"

    def test_single_row_removed(self):
        """Test with single row that is removed."""
        df_test = pd.DataFrame({"name": [""], "age": [25]})

        result = remove_empty_values(df_test, "name")

        assert result.empty

    def test_single_column(self):
        """Test with dataframe having single column."""
        df_test = pd.DataFrame({"name": ["Alice", "", "Charlie", None]})

        result = remove_empty_values(df_test, "name")

        assert result.shape == (2, 1)
        assert list(result["name"]) == ["Alice", "Charlie"]

    def test_large_dataframe(self):
        """Test with large dataframe."""
        data = {
            "id": list(range(10000)),
            "value": ["valid"] * 5000 + [""] * 5000,
        }
        df_test = pd.DataFrame(data)

        result = remove_empty_values(df_test, "value")

        assert result.shape[0] == 5000

    def test_nonexistent_column(self):
        """Test error when filter column doesn't exist."""
        df_test = pd.DataFrame({"name": ["Alice", "Bob"], "age": [25, 30]})

        with pytest.raises(KeyError):
            remove_empty_values(df_test, "nonexistent")


class TestRemoveEmptyValuesMixedEmpty:
    """Test cases for mixed empty value types."""

    def test_mixed_empty_string_and_none(self):
        """Test filtering with both empty strings and None."""
        df_test = pd.DataFrame(
            {
                "name": ["Alice", "", None, "David", ""],
                "age": [25, 30, 35, 40, 45],
            }
        )

        result = remove_empty_values(df_test, "name")

        assert result.shape[0] == 2
        assert list(result["name"]) == ["Alice", "David"]

    def test_mixed_empty_string_and_nan(self):
        """Test filtering with both empty strings and NaN."""
        df_test = pd.DataFrame(
            {
                "value": ["10", "", float("nan"), "40"],
                "label": ["A", "B", "C", "D"],
            }
        )

        result = remove_empty_values(df_test, "value")

        assert result.shape[0] == 2
        assert "10" in result["value"].to_numpy()
        assert "40" in result["value"].to_numpy()

    def test_whitespace_removed(self):
        """Test that whitespace-only strings are removed."""
        df_test = pd.DataFrame(
            {
                "name": ["Alice", " ", "  ", "David"],
                "age": [25, 30, 35, 40],
            }
        )

        result = remove_empty_values(df_test, "name")

        assert " " not in result["name"].to_numpy()

    def test_only_empty_string_removed(self):
        """Test that only truly empty strings are removed, not other falsy values."""
        df_test = pd.DataFrame(
            {
                "value": ["Alice", "", "0", "False"],
                "label": ["A", "B", "C", "D"],
            }
        )

        result = remove_empty_values(df_test, "value")

        assert result.shape[0] == 3
        assert "0" in result["value"].to_numpy()
        assert "False" in result["value"].to_numpy()


class TestRemoveEmptyValuesDataPreservation:
    """Test cases for data preservation during filtering."""

    def test_preserves_column_order(self):
        """Test that column order is preserved."""
        df_test = pd.DataFrame(
            {
                "col3": ["A", "", "C"],
                "col1": [1, 2, 3],
                "col2": ["X", "Y", "Z"],
            }
        )

        result = remove_empty_values(df_test, "col3")

        assert list(result.columns) == ["col3", "col1", "col2"]

    def test_preserves_data_types(self):
        """Test that data types are preserved."""
        df_test = pd.DataFrame(
            {
                "string_col": ["Alice", "Bob", ""],
                "int_col": [1, 2, 3],
                "float_col": [1.1, 2.2, 3.3],
                "bool_col": [True, False, True],
            }
        )

        result = remove_empty_values(df_test, "string_col")

        assert result["string_col"].dtype == "object"
        assert result["int_col"].dtype in ["int64", "int32"]
        assert result["float_col"].dtype == "float64"
        assert result["bool_col"].dtype == "bool"

    def test_preserves_other_columns(self):
        """Test that non-filter columns are preserved."""
        df_test = pd.DataFrame(
            {
                "id": [1, 2, 3, 4],
                "name": ["Alice", "", "Charlie", "David"],
                "email": ["a@test.com", "b@test.com", "c@test.com", "d@test.com"],
                "age": [25, 30, 35, 40],
            }
        )

        result = remove_empty_values(df_test, "name")

        assert "id" in result.columns
        assert "email" in result.columns
        assert "age" in result.columns
        assert result.shape[1] == 4

    def test_empty_in_other_columns_preserved(self):
        """Test that empty values in non-filter columns are preserved."""
        df_test = pd.DataFrame(
            {
                "name": ["Alice", "Bob", "Charlie"],
                "email": ["alice@test.com", "", None],
                "phone": [None, "123-456", ""],
            }
        )

        result = remove_empty_values(df_test, "name")

        assert result.shape[0] == 3
        assert result.iloc[1]["email"] == ""
        assert pd.isna(result.iloc[0]["phone"])


class TestRemoveEmptyValuesNumericColumns:
    """Test cases for filtering on numeric columns."""

    def test_filter_integer_column(self):
        """Test filtering on integer column."""
        df_test = pd.DataFrame(
            {
                "id": [1, 2, None, 4],
                "name": ["Alice", "Bob", "Charlie", "David"],
            }
        )

        result = remove_empty_values(df_test, "id")

        assert result.shape[0] == 3
        assert list(result["id"]) == [1, 2, 4]

    def test_filter_float_column(self):
        """Test filtering on float column."""
        df_test = pd.DataFrame(
            {
                "value": [1, None, 3, float("nan")],
                "label": ["A", "B", "C", "D"],
            }
        )

        result = remove_empty_values(df_test, "value")

        assert result.shape[0] == 2
        assert result.iloc[0]["value"] == 1
        assert result.iloc[1]["value"] == 3

    def test_filter_zero_not_removed(self):
        """Test that zero values are not removed."""
        df_test = pd.DataFrame(
            {
                "value": [0, 1, None, 3],
                "label": ["A", "B", "C", "D"],
            }
        )

        result = remove_empty_values(df_test, "value")

        assert result.shape[0] == 3
        assert 0 in result["value"].to_numpy()

    def test_filter_negative_values_preserved(self):
        """Test that negative values are preserved."""
        df_test = pd.DataFrame(
            {
                "value": [-1, None, -3, 4],
                "label": ["A", "B", "C", "D"],
            }
        )

        result = remove_empty_values(df_test, "value")

        assert result.shape[0] == 3
        assert -1 in result["value"].to_numpy()
        assert -3 in result["value"].to_numpy()


class TestRemoveEmptyValuesBooleanColumns:
    """Test cases for filtering on boolean columns."""

    def test_filter_boolean_column(self):
        """Test filtering on boolean column."""
        df_test = pd.DataFrame(
            {
                "active": [True, False, None, True],
                "name": ["Alice", "Bob", "Charlie", "David"],
            }
        )

        result = remove_empty_values(df_test, "active")

        assert result.shape[0] == 3
        assert list(result["name"]) == ["Alice", "Bob", "David"]

    def test_filter_false_not_removed(self):
        """Test that False boolean values are not removed."""
        df_test = pd.DataFrame(
            {
                "flag": [True, False, None, False],
                "label": ["A", "B", "C", "D"],
            }
        )

        result = remove_empty_values(df_test, "flag")

        assert result.shape[0] == 3
        assert False in result["flag"].to_numpy()


class TestRemoveEmptyValuesDatetimeColumns:
    """Test cases for filtering on datetime columns."""

    def test_filter_datetime_column(self):
        """Test filtering on datetime column."""
        df_test = pd.DataFrame(
            {
                "date": pd.to_datetime(["2023-01-01", None, "2023-03-01", "2023-04-01"]),
                "value": [1, 2, 3, 4],
            }
        )

        result = remove_empty_values(df_test, "date")

        assert result.shape[0] == 3
        assert result.iloc[0]["date"] == pd.Timestamp("2023-01-01")

    def test_filter_nat_values(self):
        """Test filtering NaT values in datetime column."""
        df_test = pd.DataFrame(
            {
                "date": pd.to_datetime(["2023-01-01", "2023-02-01", None]),
                "label": ["A", "B", "C"],
            }
        )

        result = remove_empty_values(df_test, "date")

        assert result.shape[0] == 2


class TestRemoveEmptyValuesIntegration:
    """Integration tests combining multiple scenarios."""

    def test_real_world_data_filtering(self):
        """Test filtering real-world style data."""
        df_test = pd.DataFrame(
            {
                "user_id": [1, 2, 3, 4, 5],
                "username": ["alice", "", "charlie", None, "eve"],
                "email": [
                    "alice@example.com",
                    "bob@example.com",
                    "",
                    "dave@example.com",
                    "eve@example.com",
                ],
                "score": [85.5, 90.0, 75.5, None, 95.0],
            }
        )

        result = remove_empty_values(df_test, "username")

        assert result.shape[0] == 3
        assert list(result["username"]) == ["alice", "charlie", "eve"]
        assert list(result.index) == [0, 1, 2]

    def test_multiple_filter_operations(self):
        """Test applying filter multiple times on different columns."""
        df_test = pd.DataFrame(
            {
                "col1": ["A", "", "C", "D"],
                "col2": [1, 2, None, 4],
                "col3": ["X", "Y", "Z", "W"],
            }
        )

        result1 = remove_empty_values(df_test, "col1")
        result2 = remove_empty_values(result1, "col2")

        assert result2.shape[0] == 2
        assert list(result2["col1"]) == ["A", "D"]

    def test_preserves_complex_data(self):
        """Test that complex data structures are preserved."""
        df_test = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "name": ["Alice", "", "Charlie"],
                "data": [{"key": "value1"}, {"key": "value2"}, {"key": "value3"}],
                "list_col": [[1, 2], [3, 4], [5, 6]],
            }
        )

        result = remove_empty_values(df_test, "name")

        assert result.shape[0] == 2
        assert result.iloc[0]["data"] == {"key": "value1"}
        assert result.iloc[1]["list_col"] == [5, 6]

    def test_sequential_filtering_same_column(self):
        """Test that filtering same column twice is idempotent."""
        df_test = pd.DataFrame(
            {
                "name": ["Alice", "", "Charlie", None],
                "age": [25, 30, 35, 40],
            }
        )

        result1 = remove_empty_values(df_test, "name")
        result2 = remove_empty_values(result1, "name")

        assert result1.shape == result2.shape
        pd.testing.assert_frame_equal(result1, result2)

    def test_combination_with_other_operations(self):
        """Test combining remove_empty_values with other DataFrame operations."""
        df_test = pd.DataFrame(
            {
                "category": ["A", "", "B", "A", "C"],
                "value": [10, 20, 30, 40, 50],
            }
        )

        result = remove_empty_values(df_test, "category")
        grouped = result.groupby("category")["value"].sum()

        assert grouped["A"] == 50
        assert grouped["B"] == 30
        assert grouped["C"] == 50
