"""
Comprehensive test suite for the combine_dataframes function.
Tests all scenarios, edge cases, and error handling.
"""

import pytest
import pandas as pd
from binaryrain_helper_data_processing.dataframe import combine_dataframes


class TestCombineDataframesBasic:
    """Test cases for basic dataframe combining."""

    def test_combine_basic(self):
        """Test basic combining of two dataframes."""
        df_one = pd.DataFrame({"name": ["Alice", "Bob"], "age": [25, 30]})
        df_two = pd.DataFrame({"name": ["Charlie", "David"], "age": [35, 40]})

        result = combine_dataframes(df_one, df_two)

        assert isinstance(result, pd.DataFrame)
        assert result.shape == (4, 2)
        assert list(result.columns) == ["name", "age"]
        assert list(result["name"]) == ["Alice", "Bob", "Charlie", "David"]

    def test_combine_with_sort_false(self):
        """Test combining with sort=False (default)."""
        df_one = pd.DataFrame({"col_b": [1, 2], "col_a": [3, 4]})
        df_two = pd.DataFrame({"col_a": [5, 6], "col_b": [7, 8]})

        result = combine_dataframes(df_one, df_two, sort=False)

        assert isinstance(result, pd.DataFrame)
        assert result.shape == (4, 2)

    def test_combine_with_sort_true(self):
        """Test combining with sort=True."""
        df_one = pd.DataFrame({"col_b": [1, 2], "col_a": [3, 4]})
        df_two = pd.DataFrame({"col_a": [5, 6], "col_b": [7, 8]})

        result = combine_dataframes(df_one, df_two, sort=True)

        assert isinstance(result, pd.DataFrame)
        assert result.shape == (4, 2)
        # When sorted, columns should be alphabetically ordered
        assert list(result.columns) == ["col_a", "col_b"]

    def test_combine_same_columns(self):
        """Test combining dataframes with identical columns."""
        df_one = pd.DataFrame({"name": ["Alice"], "age": [25], "city": ["NYC"]})
        df_two = pd.DataFrame({"name": ["Bob"], "age": [30], "city": ["LA"]})

        result = combine_dataframes(df_one, df_two)

        assert result.shape == (2, 3)
        assert list(result.columns) == ["name", "age", "city"]

    def test_combine_different_columns(self):
        """Test combining dataframes with different columns."""
        df_one = pd.DataFrame({"name": ["Alice"], "age": [25]})
        df_two = pd.DataFrame({"city": ["NYC"], "country": ["USA"]})

        result = combine_dataframes(df_one, df_two)

        assert result.shape == (2, 4)
        # Should have all columns from both dataframes
        assert set(result.columns) == {"name", "age", "city", "country"}

    def test_combine_overlapping_columns(self):
        """Test combining dataframes with some overlapping columns."""
        df_one = pd.DataFrame({"name": ["Alice"], "age": [25], "city": ["NYC"]})
        df_two = pd.DataFrame({"name": ["Bob"], "country": ["USA"]})

        result = combine_dataframes(df_one, df_two)

        assert result.shape == (2, 4)
        assert "name" in result.columns
        assert "age" in result.columns
        assert "city" in result.columns
        assert "country" in result.columns


class TestCombineDataframesEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_combine_empty_dataframes(self):
        """Test combining two empty dataframes."""
        df_one = pd.DataFrame()
        df_two = pd.DataFrame()

        result = combine_dataframes(df_one, df_two)

        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_combine_first_empty(self):
        """Test combining when first dataframe is empty."""
        df_one = pd.DataFrame()
        df_two = pd.DataFrame({"name": ["Alice"], "age": [25]})

        result = combine_dataframes(df_one, df_two)

        assert isinstance(result, pd.DataFrame)
        assert result.shape == (1, 2)
        assert result.iloc[0]["name"] == "Alice"

    def test_combine_second_empty(self):
        """Test combining when second dataframe is empty."""
        df_one = pd.DataFrame({"name": ["Alice"], "age": [25]})
        df_two = pd.DataFrame()

        result = combine_dataframes(df_one, df_two)

        assert isinstance(result, pd.DataFrame)
        assert result.shape == (1, 2)
        assert result.iloc[0]["name"] == "Alice"

    def test_combine_single_row_dataframes(self):
        """Test combining single row dataframes."""
        df_one = pd.DataFrame({"name": ["Alice"], "age": [25]})
        df_two = pd.DataFrame({"name": ["Bob"], "age": [30]})

        result = combine_dataframes(df_one, df_two)

        assert result.shape == (2, 2)
        assert result.iloc[0]["name"] == "Alice"
        assert result.iloc[1]["name"] == "Bob"

    def test_combine_large_dataframes(self):
        """Test combining large dataframes."""
        df_one = pd.DataFrame({"col1": range(5000), "col2": range(5000)})
        df_two = pd.DataFrame({"col1": range(5000, 10000), "col2": range(5000, 10000)})

        result = combine_dataframes(df_one, df_two)

        assert result.shape == (10000, 2)

    def test_combine_with_duplicates(self):
        """Test combining dataframes with duplicate rows."""
        df_one = pd.DataFrame({"name": ["Alice", "Bob"], "age": [25, 30]})
        df_two = pd.DataFrame({"name": ["Alice", "Charlie"], "age": [25, 35]})

        result = combine_dataframes(df_one, df_two)

        assert result.shape == (4, 2)
        # Should keep duplicates
        assert list(result["name"]) == ["Alice", "Bob", "Alice", "Charlie"]

    def test_combine_with_different_dtypes(self):
        """Test combining dataframes with different data types."""
        df_one = pd.DataFrame({"col1": [1, 2], "col2": ["a", "b"]})
        df_two = pd.DataFrame({"col1": [3, 4], "col2": ["c", "d"]})

        result = combine_dataframes(df_one, df_two)

        assert result.shape == (4, 2)
        assert result["col1"].dtype in ["int64", "int32"]
        assert result["col2"].dtype == "object"

    def test_combine_with_nan_values(self):
        """Test combining dataframes with NaN values."""
        df_one = pd.DataFrame({"name": ["Alice", "Bob"], "age": [25, None]})
        df_two = pd.DataFrame({"name": ["Charlie", None], "age": [35, 40]})

        result = combine_dataframes(df_one, df_two)

        assert result.shape == (4, 2)
        assert pd.isna(result.iloc[1]["age"])
        assert pd.isna(result.iloc[3]["name"])


class TestCombineDataframesIndexHandling:
    """Test cases for index handling."""

    def test_combine_with_default_index(self):
        """Test combining dataframes with default index."""
        df_one = pd.DataFrame({"name": ["Alice", "Bob"], "age": [25, 30]})
        df_two = pd.DataFrame({"name": ["Charlie", "David"], "age": [35, 40]})

        result = combine_dataframes(df_one, df_two)

        # Index should be preserved from original dataframes
        assert list(result.index) == [0, 1, 0, 1]

    def test_combine_with_custom_index(self):
        """Test combining dataframes with custom index."""
        df_one = pd.DataFrame({"name": ["Alice", "Bob"], "age": [25, 30]}, index=["a", "b"])
        df_two = pd.DataFrame({"name": ["Charlie", "David"], "age": [35, 40]}, index=["c", "d"])

        result = combine_dataframes(df_one, df_two)

        assert result.shape == (4, 2)
        assert list(result.index) == ["a", "b", "c", "d"]

    def test_combine_with_overlapping_index(self):
        """Test combining dataframes with overlapping index values."""
        df_one = pd.DataFrame({"name": ["Alice", "Bob"], "age": [25, 30]}, index=[0, 1])
        df_two = pd.DataFrame({"name": ["Charlie", "David"], "age": [35, 40]}, index=[0, 1])

        result = combine_dataframes(df_one, df_two)

        assert result.shape == (4, 2)
        # Both dataframes have index [0, 1], so result should have [0, 1, 0, 1]
        assert list(result.index) == [0, 1, 0, 1]

    def test_combine_with_multi_index(self):
        """Test combining dataframes with multi-index."""
        df_one = pd.DataFrame({"value": [1, 2]})
        df_one.index = pd.MultiIndex.from_tuples([("a", 1), ("a", 2)])

        df_two = pd.DataFrame({"value": [3, 4]})
        df_two.index = pd.MultiIndex.from_tuples([("b", 1), ("b", 2)])

        result = combine_dataframes(df_one, df_two)

        assert result.shape == (4, 1)
        assert isinstance(result.index, pd.MultiIndex)


class TestCombineDataframesColumnTypes:
    """Test cases for different column types."""

    def test_combine_with_numeric_columns(self):
        """Test combining dataframes with numeric columns."""
        df_one = pd.DataFrame({"int_col": [1, 2], "float_col": [1.1, 2.2]})
        df_two = pd.DataFrame({"int_col": [3, 4], "float_col": [3.3, 4.4]})

        result = combine_dataframes(df_one, df_two)

        assert result.shape == (4, 2)
        assert result["int_col"].dtype in ["int64", "int32"]
        assert result["float_col"].dtype == "float64"

    def test_combine_with_datetime_columns(self):
        """Test combining dataframes with datetime columns."""
        df_one = pd.DataFrame(
            {"date": [pd.Timestamp("2023-01-01"), pd.Timestamp("2023-01-02")], "value": [1, 2]}
        )
        df_two = pd.DataFrame(
            {"date": [pd.Timestamp("2023-01-03"), pd.Timestamp("2023-01-04")], "value": [3, 4]}
        )

        result = combine_dataframes(df_one, df_two)

        assert result.shape == (4, 2)
        assert pd.api.types.is_datetime64_any_dtype(result["date"])

    def test_combine_with_boolean_columns(self):
        """Test combining dataframes with boolean columns."""
        df_one = pd.DataFrame({"flag": [True, False], "value": [1, 2]})
        df_two = pd.DataFrame({"flag": [True, True], "value": [3, 4]})

        result = combine_dataframes(df_one, df_two)

        assert result.shape == (4, 2)
        assert result["flag"].dtype == "bool"

    def test_combine_with_categorical_columns(self):
        """Test combining dataframes with categorical columns."""
        df_one = pd.DataFrame({"category": pd.Categorical(["A", "B"]), "value": [1, 2]})
        df_two = pd.DataFrame({"category": pd.Categorical(["C", "D"]), "value": [3, 4]})

        result = combine_dataframes(df_one, df_two)

        assert result.shape == (4, 2)

    def test_combine_with_mixed_column_types(self):
        """Test combining dataframes with mixed column types."""
        df_one = pd.DataFrame(
            {
                "string_col": ["a", "b"],
                "int_col": [1, 2],
                "float_col": [1.1, 2.2],
                "bool_col": [True, False],
            }
        )
        df_two = pd.DataFrame(
            {
                "string_col": ["c", "d"],
                "int_col": [3, 4],
                "float_col": [3.3, 4.4],
                "bool_col": [False, True],
            }
        )

        result = combine_dataframes(df_one, df_two)

        assert result.shape == (4, 4)


class TestCombineDataframesErrorHandling:
    """Test cases for error handling."""

    def test_combine_none_first_dataframe(self):
        """Test with None as first dataframe."""
        df_two = pd.DataFrame({"name": ["Alice"], "age": [25]})

        with pytest.raises(ValueError, match="No dataframe provided for df_one"):
            combine_dataframes(None, df_two)

    def test_combine_none_second_dataframe(self):
        """Test with None as second dataframe."""
        df_one = pd.DataFrame({"name": ["Alice"], "age": [25]})

        with pytest.raises(ValueError, match="No dataframe provided for df_one"):
            combine_dataframes(df_one, None)

    def test_combine_both_none(self):
        """Test with both dataframes as None."""
        with pytest.raises(ValueError, match="No dataframe provided for df_one"):
            combine_dataframes(None, None)

    def test_combine_invalid_first_type(self):
        """Test with invalid type for first dataframe."""
        df_two = pd.DataFrame({"name": ["Alice"], "age": [25]})

        with pytest.raises(ValueError, match="No dataframe provided for df_one"):
            combine_dataframes("not a dataframe", df_two)

    def test_combine_invalid_second_type(self):
        """Test with invalid type for second dataframe."""
        df_one = pd.DataFrame({"name": ["Alice"], "age": [25]})

        with pytest.raises(ValueError, match="No dataframe provided for df_one"):
            combine_dataframes(df_one, "not a dataframe")

    def test_combine_dict_instead_of_dataframe(self):
        """Test passing dict instead of dataframe."""
        df_one = pd.DataFrame({"name": ["Alice"], "age": [25]})
        dict_data = {"name": ["Bob"], "age": [30]}

        with pytest.raises(ValueError, match="No dataframe provided for df_one"):
            combine_dataframes(df_one, dict_data)

    def test_combine_list_instead_of_dataframe(self):
        """Test passing list instead of dataframe."""
        df_one = pd.DataFrame({"name": ["Alice"], "age": [25]})
        list_data = [["Bob", 30]]

        with pytest.raises(ValueError, match="No dataframe provided for df_one"):
            combine_dataframes(df_one, list_data)


class TestCombineDataframesSpecialScenarios:
    """Test special scenarios and use cases."""

    def test_combine_with_unicode_characters(self):
        """Test combining dataframes with unicode characters."""
        df_one = pd.DataFrame({"name": ["Alice", "José"], "greeting": ["Hello", "Hola"]})
        df_two = pd.DataFrame({"name": ["李明", "Müller"], "greeting": ["你好", "Guten Tag"]})

        result = combine_dataframes(df_one, df_two)

        assert result.shape == (4, 2)
        assert result.iloc[1]["name"] == "José"
        assert result.iloc[2]["greeting"] == "你好"

    def test_combine_with_special_characters(self):
        """Test combining dataframes with special characters."""
        df_one = pd.DataFrame({"name": ["Test, Inc.", 'Quote"d'], "value": [100, 200]})
        df_two = pd.DataFrame({"name": ["New\nLine", "Tab\there"], "value": [300, 400]})

        result = combine_dataframes(df_one, df_two)

        assert result.shape == (4, 2)
        assert result.iloc[0]["name"] == "Test, Inc."

    def test_combine_preserves_column_order(self):
        """Test that combining preserves column order from first dataframe."""
        df_one = pd.DataFrame({"col_c": [1], "col_a": [2], "col_b": [3]})
        df_two = pd.DataFrame({"col_c": [4], "col_a": [5], "col_b": [6]})

        result = combine_dataframes(df_one, df_two, sort=False)

        assert list(result.columns) == ["col_c", "col_a", "col_b"]

    def test_combine_with_different_row_counts(self):
        """Test combining dataframes with different row counts."""
        df_one = pd.DataFrame({"name": ["Alice"], "age": [25]})
        df_two = pd.DataFrame({"name": ["Bob", "Charlie", "David"], "age": [30, 35, 40]})

        result = combine_dataframes(df_one, df_two)

        assert result.shape == (4, 2)

    def test_combine_multiple_times(self):
        """Test combining result with another dataframe."""
        df_one = pd.DataFrame({"name": ["Alice"], "age": [25]})
        df_two = pd.DataFrame({"name": ["Bob"], "age": [30]})
        df_three = pd.DataFrame({"name": ["Charlie"], "age": [35]})

        result_intermediate = combine_dataframes(df_one, df_two)
        result_final = combine_dataframes(result_intermediate, df_three)

        assert result_final.shape == (3, 2)
        assert list(result_final["name"]) == ["Alice", "Bob", "Charlie"]


class TestCombineDataframesIntegration:
    """Integration tests combining multiple scenarios."""

    def test_combine_then_filter(self):
        """Test combining dataframes then filtering result."""
        df_one = pd.DataFrame({"name": ["Alice", "Bob"], "age": [25, 30]})
        df_two = pd.DataFrame({"name": ["Charlie", "David"], "age": [35, 40]})

        result = combine_dataframes(df_one, df_two)
        filtered = result[result["age"] > 30]

        assert filtered.shape == (2, 2)
        assert list(filtered["name"]) == ["Charlie", "David"]

    def test_combine_then_sort(self):
        """Test combining dataframes then sorting result."""
        df_one = pd.DataFrame({"name": ["Charlie", "Alice"], "age": [35, 25]})
        df_two = pd.DataFrame({"name": ["David", "Bob"], "age": [40, 30]})

        result = combine_dataframes(df_one, df_two)
        sorted_result = result.sort_values("age")

        assert sorted_result.iloc[0]["name"] == "Alice"
        assert sorted_result.iloc[-1]["name"] == "David"

    def test_combine_then_aggregate(self):
        """Test combining dataframes then aggregating."""
        df_one = pd.DataFrame({"category": ["A", "B"], "value": [10, 20]})
        df_two = pd.DataFrame({"category": ["A", "B"], "value": [15, 25]})

        result = combine_dataframes(df_one, df_two)
        grouped = result.groupby("category")["value"].sum()

        assert grouped["A"] == 25
        assert grouped["B"] == 45

    def test_combine_then_reset_index(self):
        """Test combining dataframes then resetting index."""
        df_one = pd.DataFrame({"name": ["Alice", "Bob"], "age": [25, 30]})
        df_two = pd.DataFrame({"name": ["Charlie", "David"], "age": [35, 40]})

        result = combine_dataframes(df_one, df_two)
        reset_result = result.reset_index(drop=True)

        assert list(reset_result.index) == [0, 1, 2, 3]

    def test_combine_symmetric(self):
        """Test that combining is order-dependent for structure."""
        df_one = pd.DataFrame({"name": ["Alice"], "age": [25]})
        df_two = pd.DataFrame({"name": ["Bob"], "age": [30]})

        result_1 = combine_dataframes(df_one, df_two)
        result_2 = combine_dataframes(df_two, df_one)

        # Both should have same data, but order may differ
        assert result_1.shape == result_2.shape
        assert set(result_1["name"]) == set(result_2["name"])
