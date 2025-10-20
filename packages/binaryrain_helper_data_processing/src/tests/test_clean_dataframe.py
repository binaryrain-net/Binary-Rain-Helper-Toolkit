"""
Comprehensive test suite for the clean_dataframe function.
Tests all scenarios, edge cases, and data cleaning operations.
"""

import pandas as pd
from binaryrain_helper_data_processing.dataframe import clean_dataframe


class TestCleanDataframeBasic:
    """Test cases for basic dataframe cleaning scenarios."""

    def test_clean_basic_dataframe(self):
        """Test cleaning a basic dataframe without missing values."""
        df_test = pd.DataFrame(
            {
                "name": ["Alice", "Bob", "Charlie"],
                "age": [25, 30, 35],
                "city": ["New York", "Los Angeles", "Chicago"],
            }
        )

        result = clean_dataframe(df_test)

        assert isinstance(result, pd.DataFrame)
        assert result.shape == (3, 3)
        assert list(result.columns) == ["name", "age", "city"]
        assert result.iloc[0]["name"] == "Alice"

    def test_clean_removes_nan_strings(self):
        """Test that string 'nan' values are removed."""
        df_test = pd.DataFrame({"name": ["Alice", "nan", "Charlie"], "age": [25, 30, 35]})

        result = clean_dataframe(df_test)

        assert result.shape[0] == 2
        assert "Alice" in result["name"].to_numpy()
        assert "Charlie" in result["name"].to_numpy()
        assert "nan" not in result["name"].to_numpy()

    def test_clean_removes_empty_strings(self):
        """Test that empty string values are removed."""
        df_test = pd.DataFrame({"name": ["Alice", "", "Charlie"], "age": [25, 30, 35]})

        result = clean_dataframe(df_test)

        assert result.shape[0] == 2
        assert "Alice" in result["name"].to_numpy()
        assert "Charlie" in result["name"].to_numpy()

    def test_clean_removes_na_values(self):
        """Test that pd.NA values are removed."""
        df_test = pd.DataFrame({"name": ["Alice", pd.NA, "Charlie"], "age": [25, 30, 35]})

        result = clean_dataframe(df_test)

        assert result.shape[0] == 2
        assert "Alice" in result["name"].to_numpy()
        assert "Charlie" in result["name"].to_numpy()

    def test_clean_removes_none_values(self):
        """Test that None values are removed."""
        df_test = pd.DataFrame({"name": ["Alice", None, "Charlie"], "age": [25, 30, 35]})

        result = clean_dataframe(df_test)

        assert result.shape[0] == 2
        assert "Alice" in result["name"].to_numpy()
        assert "Charlie" in result["name"].to_numpy()


class TestCleanDataframeDuplicates:
    """Test cases for duplicate removal."""

    def test_clean_multiple_duplicates(self):
        """Test removal of multiple duplicate rows."""
        df_test = pd.DataFrame(
            {
                "name": ["Alice", "Bob", "Alice", "Bob", "Charlie"],
                "age": [25, 30, 25, 30, 35],
            }
        )

        result = clean_dataframe(df_test)

        assert result.shape[0] == 5
        assert "Alice" in result["name"].to_numpy()
        assert "Bob" in result["name"].to_numpy()
        assert "Charlie" in result["name"].to_numpy()

    def test_clean_all_duplicates(self):
        """Test when all rows except one are duplicates."""
        df_test = pd.DataFrame(
            {"name": ["Alice", "Alice", "Alice", "Alice"], "age": [25, 25, 25, 25]}
        )

        result = clean_dataframe(df_test)

        assert result.shape[0] == 4
        assert result.iloc[0]["name"] == "Alice"

    def test_clean_no_duplicates(self):
        """Test when there are no duplicate rows."""
        df_test = pd.DataFrame({"name": ["Alice", "Bob", "Charlie"], "age": [25, 30, 35]})

        result = clean_dataframe(df_test)

        assert result.shape[0] == 3


class TestCleanDataframeIndexReset:
    """Test cases for index reset after cleaning."""

    def test_index_reset_after_dropna(self):
        """Test that index is reset after dropping NA values."""
        df_test = pd.DataFrame(
            {"name": ["Alice", None, "Charlie", "David"], "age": [25, 30, 35, 40]}
        )

        result = clean_dataframe(df_test)

        assert list(result.index) == [0, 1, 2]
        assert result.iloc[0]["name"] == "Alice"
        assert result.iloc[1]["name"] == "Charlie"
        assert result.iloc[2]["name"] == "David"

    def test_index_reset_combined_operations(self):
        """Test that index is reset after all cleaning operations."""
        df_test = pd.DataFrame(
            {
                "name": ["Alice", "nan", "Bob", "Alice", "Charlie"],
                "age": [25, 30, 35, 25, 40],
            },
            index=[5, 10, 15, 20, 25],
        )

        result = clean_dataframe(df_test)

        assert result.shape[0] == 4
        assert list(result["name"]) == ["Alice", "Bob", "Alice", "Charlie"]


class TestCleanDataframeMultipleColumns:
    """Test cases for cleaning dataframes with multiple columns."""

    def test_clean_missing_in_different_columns(self):
        """Test cleaning when missing values are in different columns."""
        df_test = pd.DataFrame(
            {
                "name": ["Alice", "nan", "Charlie"],
                "age": [25, 30, None],
                "city": ["New York", "", "Chicago"],
            }
        )

        result = clean_dataframe(df_test)

        assert result.shape[0] == 1
        assert result.iloc[0]["name"] == "Alice"

    def test_clean_preserves_valid_rows(self):
        """Test that rows with all valid values are preserved."""
        df_test = pd.DataFrame(
            {
                "col1": ["A", "nan", "C", "D"],
                "col2": [1, 2, None, 4],
                "col3": ["X", "Y", "Z", "W"],
            }
        )

        result = clean_dataframe(df_test)

        assert result.shape[0] == 2
        assert result.iloc[0]["col1"] == "A"
        assert result.iloc[1]["col1"] == "D"

    def test_clean_multiple_column_types(self):
        """Test cleaning dataframe with multiple column types."""
        df_test = pd.DataFrame(
            {
                "string_col": ["Alice", "Bob", "nan"],
                "int_col": [1, 2, 3],
                "float_col": [1.1, 2.2, 3.3],
                "bool_col": [True, False, True],
            }
        )

        result = clean_dataframe(df_test)

        assert result.shape[0] == 2
        assert result["string_col"].iloc[0] == "Alice"
        assert result["int_col"].iloc[1] == 2


class TestCleanDataframeEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_clean_empty_dataframe(self):
        """Test cleaning an empty dataframe."""
        df_test = pd.DataFrame()

        result = clean_dataframe(df_test)

        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_clean_single_row(self):
        """Test cleaning a dataframe with single row."""
        df_test = pd.DataFrame({"name": ["Alice"], "age": [25]})

        result = clean_dataframe(df_test)

        assert result.shape[0] == 1
        assert result.iloc[0]["name"] == "Alice"

    def test_clean_single_column(self):
        """Test cleaning a dataframe with single column."""
        df_test = pd.DataFrame({"name": ["Alice", "nan", "Charlie"]})

        result = clean_dataframe(df_test)

        assert result.shape == (2, 1)
        assert "Alice" in result["name"].to_numpy()

    def test_clean_all_rows_removed(self):
        """Test when all rows are removed due to missing values."""
        df_test = pd.DataFrame({"name": ["nan", None, ""], "age": [None, pd.NA, None]})

        result = clean_dataframe(df_test)

        assert isinstance(result, pd.DataFrame)
        assert result.empty
        assert list(result.columns) == ["name", "age"]

    def test_clean_large_dataframe(self):
        """Test cleaning a large dataframe."""
        data = {
            "col1": ["valid"] * 5000 + ["nan"] * 5000,
            "col2": list(range(10000)),
        }
        df_test = pd.DataFrame(data)

        result = clean_dataframe(df_test)

        assert result.shape[0] == 5000

    def test_clean_single_duplicate_row(self):
        """Test cleaning when only one row is duplicated."""
        df_test = pd.DataFrame({"name": ["Alice", "Alice"], "age": [25, 25]})

        result = clean_dataframe(df_test)

        assert result.shape[0] == 2


class TestCleanDataframeNanVariations:
    """Test cases for different variations of NaN/missing values."""

    def test_clean_mixed_nan_types(self):
        """Test cleaning with mixed NaN types (string 'nan', None, pd.NA)."""
        df_test = pd.DataFrame(
            {
                "col1": ["Alice", "nan", "Charlie", None, "Eve"],
                "col2": [1, 2, 3, 4, 5],
            }
        )

        result = clean_dataframe(df_test)

        assert result.shape[0] == 3
        assert "Alice" in result["col1"].to_numpy()
        assert "Charlie" in result["col1"].to_numpy()
        assert "Eve" in result["col1"].to_numpy()

    def test_clean_nan_in_numeric_column(self):
        """Test cleaning NaN values in numeric columns."""
        df_test = pd.DataFrame({"name": ["Alice", "Bob", "Charlie"], "age": [25, None, 35]})

        result = clean_dataframe(df_test)

        assert result.shape[0] == 2
        assert result.iloc[0]["name"] == "Alice"
        assert result.iloc[1]["name"] == "Charlie"

    def test_clean_empty_string_whitespace(self):
        """Test that empty strings are removed but whitespace strings remain."""
        df_test = pd.DataFrame({"name": ["Alice", "", " ", "Charlie"], "age": [25, 30, 35, 40]})

        result = clean_dataframe(df_test)

        assert result.shape[0] == 3
        assert " " in result["name"].to_numpy()

    def test_clean_nan_string_case_sensitive(self):
        """Test that only lowercase 'nan' string is replaced."""
        df_test = pd.DataFrame(
            {"name": ["Alice", "nan", "NAN", "Nan", "Bob"], "age": [25, 30, 35, 40, 45]}
        )

        result = clean_dataframe(df_test)

        assert result.shape[0] == 4
        assert "nan" not in result["name"].to_numpy()
        assert "NAN" in result["name"].to_numpy()
        assert "Nan" in result["name"].to_numpy()


class TestCleanDataframeDataPreservation:
    """Test cases for data preservation during cleaning."""

    def test_clean_preserves_column_order(self):
        """Test that column order is preserved after cleaning."""
        df_test = pd.DataFrame(
            {
                "col3": ["A", "B", "C"],
                "col1": [1, 2, 3],
                "col2": ["X", "Y", "Z"],
            }
        )

        result = clean_dataframe(df_test)

        assert list(result.columns) == ["col3", "col1", "col2"]

    def test_clean_preserves_data_types(self):
        """Test that data types are preserved after cleaning."""
        df_test = pd.DataFrame(
            {
                "string_col": ["Alice", "Bob"],
                "int_col": [1, 2],
                "float_col": [1.1, 2.2],
                "bool_col": [True, False],
            }
        )

        result = clean_dataframe(df_test)

        assert result["string_col"].dtype == "object"
        assert result["int_col"].dtype in ["int64", "int32"]
        assert result["float_col"].dtype == "float64"
        assert result["bool_col"].dtype == "bool"

    def test_clean_preserves_column_names(self):
        """Test that column names are preserved after cleaning."""
        df_test = pd.DataFrame(
            {
                "First Name": ["Alice", "Bob"],
                "Age (years)": [25, 30],
                "City/Town": ["New York", "Los Angeles"],
            }
        )

        result = clean_dataframe(df_test)

        assert "First Name" in result.columns
        assert "Age (years)" in result.columns
        assert "City/Town" in result.columns

    def test_original_dataframe_modified(self):
        """Test that original dataframe is modified (pandas behavior)."""
        df_test = pd.DataFrame({"name": ["Alice", "", "Alice"], "age": [25, 30, 25]})
        original_shape = df_test.shape

        result = clean_dataframe(df_test)

        assert result.shape != original_shape


class TestCleanDataframeCombinedOperations:
    """Test cases for combined cleaning operations."""

    def test_clean_nan_and_duplicates(self):
        """Test cleaning both NaN values and duplicates."""
        df_test = pd.DataFrame(
            {
                "name": ["Alice", "nan", "Bob", "Alice", "Charlie"],
                "age": [25, 30, 35, 25, 40],
            }
        )

        result = clean_dataframe(df_test)

        assert result.shape[0] == 4
        assert "nan" not in result["name"].to_numpy()
        assert len(result[result["name"] == "Alice"]) == 2

    def test_clean_empty_strings_and_duplicates(self):
        """Test cleaning empty strings and duplicates."""
        df_test = pd.DataFrame(
            {
                "name": ["Alice", "", "Bob", "Alice", "Charlie"],
                "age": [25, 30, 35, 25, 40],
            }
        )

        result = clean_dataframe(df_test)

        assert result.shape[0] == 4
        assert "" not in result["name"].to_numpy()

    def test_clean_multiple_issues_per_row(self):
        """Test cleaning rows with multiple types of issues."""
        df_test = pd.DataFrame(
            {
                "col1": ["Alice", "nan", "Bob", "Alice"],
                "col2": [25, None, 35, 25],
                "col3": ["X", "Y", "", "X"],
            }
        )

        result = clean_dataframe(df_test)

        assert result.shape[0] == 2
        assert result.iloc[0]["col1"] == "Alice"

    def test_clean_sequential_operations(self):
        """Test that cleaning operations happen in correct sequence."""
        df_test = pd.DataFrame(
            {
                "name": ["Alice", "nan", "", "Bob", "Alice", None],
                "age": [25, 30, 35, 40, 25, 50],
            }
        )

        result = clean_dataframe(df_test)

        assert result.shape[0] == 3


class TestCleanDataframeSpecialCases:
    """Test cases for special data scenarios."""

    def test_clean_numeric_data_only(self):
        """Test cleaning dataframe with only numeric columns."""
        df_test = pd.DataFrame({"col1": [1, 2, None, 4], "col2": [10.5, 20.5, 30.5, 40.5]})

        result = clean_dataframe(df_test)

        assert result.shape[0] == 3

    def test_clean_boolean_data(self):
        """Test cleaning dataframe with boolean columns."""
        df_test = pd.DataFrame(
            {"flag1": [True, False, True, None], "flag2": [False, True, False, True]}
        )

        result = clean_dataframe(df_test)

        assert list(result["flag1"].to_numpy()) == [True, False, True]
        assert list(result["flag2"].to_numpy()) == [False, True, False]

    def test_clean_datetime_data(self):
        """Test cleaning dataframe with datetime columns."""
        df_test = pd.DataFrame(
            {
                "date": pd.to_datetime(["2023-01-01", "2023-02-01", None, "2023-04-01"]),
                "value": [1, 2, 3, 4],
            }
        )

        result = clean_dataframe(df_test)

        assert result.shape[0] == 3

    def test_clean_mixed_types_complex(self):
        """Test cleaning complex dataframe with mixed types."""
        df_test = pd.DataFrame(
            {
                "id": [1, 2, 3, 4, 5],
                "name": ["Alice", "nan", "Charlie", "", "Eve"],
                "date": pd.to_datetime(
                    ["2023-01-01", "2023-02-01", None, "2023-04-01", "2023-05-01"]
                ),
                "value": [10.5, None, 30.5, 40.5, 50.5],
                "active": [True, False, True, True, None],
            }
        )

        result = clean_dataframe(df_test)

        assert result.shape[0] == 1
        assert result.iloc[0]["name"] == "Alice"


class TestCleanDataframeIntegration:
    """Integration tests combining multiple scenarios."""

    def test_real_world_data_cleaning(self):
        """Test cleaning real-world style messy data (removes rows with missing/invalid values)."""
        df_test = pd.DataFrame(
            {
                "user_id": [1, 2, 3, 4, 5, 6, 7],
                "username": ["alice", "nan", "charlie", "alice", "", "frank", "george"],
                "email": [
                    "alice@example.com",
                    "bob@example.com",
                    None,
                    "alice@example.com",
                    "eve@example.com",
                    "frank@example.com",
                    "george@example.com",
                ],
                "score": [85.5, 90.0, 75.5, 85.5, None, 95.0, 88.0],
            }
        )

        result = clean_dataframe(df_test)

        assert result.shape[0] == 4
        assert list(result.index) == [0, 1, 2, 3]
        assert "nan" not in result["username"].to_numpy()
        assert "" not in result["username"].to_numpy()
        assert None not in result["email"].to_numpy()
        assert None not in result["score"].to_numpy()

    def test_dataframe_after_multiple_cleans(self):
        """Test applying clean_dataframe multiple times."""
        df_test = pd.DataFrame({"name": ["Alice", "Bob", "Alice"], "age": [25, 30, 25]})

        result1 = clean_dataframe(df_test)
        result2 = clean_dataframe(result1)

        assert result1.shape == result2.shape
        pd.testing.assert_frame_equal(result1, result2)

    def test_clean_preserves_unique_valid_data(self):
        """Test that all unique valid data is preserved."""
        df_test = pd.DataFrame(
            {
                "category": ["A", "B", "nan", "C", "", "D", None, "E"],
                "value": [1, 2, 3, 4, 5, 6, 7, 8],
            }
        )

        result = clean_dataframe(df_test)

        assert result.shape[0] == 5
        assert set(result["category"]) == {"A", "B", "C", "D", "E"}
