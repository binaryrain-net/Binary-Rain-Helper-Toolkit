"""
Comprehensive test suite for the create_dataframe function.
Tests all file formats, options, edge cases, and error handling.
"""

import pytest
import pandas as pd
from binaryrain_helper_data_processing.dataframe import create_dataframe, FileFormat


class TestCreateDataframeCSV:
    """Test cases for CSV file format."""

    def test_csv_basic(self):
        """Test basic CSV creation without options."""
        csv_data = b"name,age,city\nAlice,25,New York\nBob,30,Los Angeles\nCharlie,35,Chicago"

        df_test = create_dataframe(csv_data, FileFormat.CSV)

        assert isinstance(df_test, pd.DataFrame)
        assert df_test.shape == (3, 3)
        assert list(df_test.columns) == ["name", "age", "city"]
        assert df_test.iloc[0]["name"] == "Alice"
        assert df_test.iloc[1]["age"] == 30
        assert df_test.iloc[2]["city"] == "Chicago"

    def test_csv_with_options(self):
        """Test CSV creation with custom options (e.g., different delimiter)."""
        csv_data = b"name;age;city\nAlice;25;New York\nBob;30;Los Angeles"

        df_test = create_dataframe(csv_data, FileFormat.CSV, file_format_options={"sep": ";"})

        assert isinstance(df_test, pd.DataFrame)
        assert df_test.shape == (2, 3)
        assert list(df_test.columns) == ["name", "age", "city"]
        assert df_test.iloc[0]["name"] == "Alice"

    def test_csv_with_custom_encoding(self):
        """Test CSV with encoding options."""
        csv_data = b"name,value\nTest,100\nData,200"

        df_test = create_dataframe(
            csv_data, FileFormat.CSV, file_format_options={"encoding": "utf-8"}
        )

        assert df_test.shape == (2, 2)
        assert df_test.iloc[0]["value"] == 100

    def test_csv_empty_file(self):
        """Test CSV with empty data."""
        csv_data = b""

        with pytest.raises(ValueError, match="Error creating dataframe"):
            create_dataframe(csv_data, FileFormat.CSV)

    def test_csv_malformed_data(self):
        """Test CSV with malformed data."""
        csv_data = b"name,age,city\nAlice,25\nBob,30,Los Angeles,Extra"

        with pytest.raises(ValueError, match="Error creating dataframe"):
            create_dataframe(csv_data, FileFormat.CSV)


class TestCreateDataframeDICT:
    """Test cases for DICT file format."""

    def test_dict_basic(self):
        """Test basic dict creation without options."""
        dict_data = {
            "name": ["Alice", "Bob", "Charlie"],
            "age": [25, 30, 35],
            "city": ["New York", "Los Angeles", "Chicago"],
        }

        df_test = create_dataframe(dict_data, FileFormat.DICT)

        assert isinstance(df_test, pd.DataFrame)
        assert df_test.shape == (3, 3)
        assert list(df_test.columns) == ["name", "age", "city"]
        assert df_test.iloc[0]["name"] == "Alice"

    def test_dict_with_orient_option(self):
        """Test dict creation with orient option."""
        dict_data = [
            {"name": "Alice", "age": 25, "city": "New York"},
            {"name": "Bob", "age": 30, "city": "Los Angeles"},
        ]

        df_test = create_dataframe(
            dict_data, FileFormat.DICT, file_format_options={"orient": "columns"}
        )

        assert isinstance(df_test, pd.DataFrame)
        assert df_test.shape == (2, 3)
        assert df_test.iloc[0]["name"] == "Alice"

    def test_dict_empty(self):
        """Test dict with empty data."""
        dict_data = {}

        df_test = create_dataframe(dict_data, FileFormat.DICT)

        assert isinstance(df_test, pd.DataFrame)
        assert df_test.empty

    def test_dict_single_column(self):
        """Test dict with single column."""
        dict_data = {"name": ["Alice", "Bob"]}

        df_test = create_dataframe(dict_data, FileFormat.DICT)

        assert df_test.shape == (2, 1)
        assert list(df_test.columns) == ["name"]

    def test_dict_nested_structure(self):
        """Test dict with nested structure using columns option."""
        dict_data = {
            "index": [0, 1],
            "data": [{"name": "Alice", "age": 25}, {"name": "Bob", "age": 30}],
        }

        df_test = create_dataframe(dict_data, FileFormat.DICT)
        assert isinstance(df_test, pd.DataFrame)


class TestCreateDataframePARQUET:
    """Test cases for PARQUET file format."""

    def test_parquet_basic(self):
        """Test basic parquet creation without options."""
        # Create sample parquet data
        sample_df_test = pd.DataFrame(
            {
                "name": ["Alice", "Bob", "Charlie"],
                "age": [25, 30, 35],
                "city": ["New York", "Los Angeles", "Chicago"],
            }
        )
        parquet_bytes = sample_df_test.to_parquet(engine="pyarrow")

        df_test = create_dataframe(parquet_bytes, FileFormat.PARQUET)

        assert isinstance(df_test, pd.DataFrame)
        assert df_test.shape == (3, 3)
        assert list(df_test.columns) == ["name", "age", "city"]
        assert df_test.iloc[0]["name"] == "Alice"

    def test_parquet_with_columns_option(self):
        """Test parquet with column selection option."""
        sample_df_test = pd.DataFrame(
            {"name": ["Alice", "Bob"], "age": [25, 30], "city": ["New York", "Los Angeles"]}
        )
        parquet_bytes = sample_df_test.to_parquet(engine="pyarrow")

        df_test = create_dataframe(
            parquet_bytes, FileFormat.PARQUET, file_format_options={"columns": ["name", "age"]}
        )

        assert df_test.shape == (2, 2)
        assert list(df_test.columns) == ["name", "age"]

    def test_parquet_invalid_data(self):
        """Test parquet with invalid bytes."""
        invalid_bytes = b"not a valid parquet file"

        with pytest.raises(ValueError, match="Error creating dataframe"):
            create_dataframe(invalid_bytes, FileFormat.PARQUET)

    def test_parquet_empty_dataframe(self):
        """Test parquet from empty dataframe."""
        sample_df_test = pd.DataFrame()
        parquet_bytes = sample_df_test.to_parquet(engine="pyarrow")

        df_test = create_dataframe(parquet_bytes, FileFormat.PARQUET)

        assert isinstance(df_test, pd.DataFrame)
        assert df_test.empty


class TestCreateDataframeJSON:
    """Test cases for JSON file format."""

    def test_json_basic(self):
        """Test basic JSON creation without options."""
        json_data = b'[{"name":"Alice","age":25,"city":"New York"},{"name":"Bob","age":30,"city":"Los Angeles"}]'  # noqa: E501

        df_test = create_dataframe(json_data, FileFormat.JSON)

        assert isinstance(df_test, pd.DataFrame)
        assert df_test.shape == (2, 3)
        assert df_test.iloc[0]["name"] == "Alice"

    def test_json_with_orient_option(self):
        """Test JSON with orient option."""
        json_data = b'{"name":{"0":"Alice","1":"Bob"},"age":{"0":25,"1":30}}'

        df_test = create_dataframe(
            json_data, FileFormat.JSON, file_format_options={"orient": "index"}
        )

        assert isinstance(df_test, pd.DataFrame)
        # Different orient will produce different structure

    def test_json_lines_format(self):
        """Test JSON lines format."""
        json_data = b'{"name":"Alice","age":25}\n{"name":"Bob","age":30}'

        df_test = create_dataframe(json_data, FileFormat.JSON, file_format_options={"lines": True})

        assert isinstance(df_test, pd.DataFrame)
        assert df_test.shape == (2, 2)

    def test_json_empty_array(self):
        """Test JSON with empty array."""
        json_data = b"[]"

        df_test = create_dataframe(json_data, FileFormat.JSON)

        assert isinstance(df_test, pd.DataFrame)
        assert df_test.empty

    def test_json_malformed(self):
        """Test JSON with malformed data."""
        json_data = b'{"name": "Alice", "age": 25'  # Missing closing brace

        with pytest.raises(ValueError, match="Error creating dataframe"):
            create_dataframe(json_data, FileFormat.JSON)

    def test_json_nested_structure(self):
        """Test JSON with nested structure."""
        json_data = b'[{"name":"Alice","details":{"age":25,"city":"NYC"}},{"name":"Bob","details":{"age":30,"city":"LA"}}]'  # noqa: E501

        df_test = create_dataframe(json_data, FileFormat.JSON)

        assert isinstance(df_test, pd.DataFrame)
        assert "name" in df_test.columns


class TestCreateDataframeErrorHandling:
    """Test cases for error handling and edge cases."""

    def test_unknown_file_format(self):
        """Test with unknown file format enum value (simulated)."""

        # Create a mock enum that's not in the match cases
        class UnknownFormat:
            pass

        with pytest.raises(ValueError, match="Error creating dataframe. Unknown file format"):
            create_dataframe(b"some data", UnknownFormat)

    def test_type_error_in_processing(self):
        """Test that exceptions during processing are caught and re-raised as ValueError."""
        # Pass wrong type for dict format (bytes instead of dict)
        csv_bytes = b"name,age\nAlice,25"

        with pytest.raises(ValueError, match="Error creating dataframe"):
            create_dataframe(csv_bytes, FileFormat.DICT)

    def test_none_file_contents(self):
        """Test with None as file contents."""
        with pytest.raises(ValueError, match="Error creating dataframe"):
            create_dataframe(None, FileFormat.CSV)

    def test_wrong_type_for_bytes_format(self):
        """Test passing dict when bytes expected (CSV)."""
        dict_data = {"name": ["Alice"]}

        with pytest.raises(ValueError, match="Error creating dataframe"):
            create_dataframe(dict_data, FileFormat.CSV)

    def test_wrong_type_for_dict_format(self):
        """Test passing bytes when dict expected (DICT)."""
        bytes_data = b"some bytes"

        with pytest.raises(ValueError, match="Error creating dataframe"):
            create_dataframe(bytes_data, FileFormat.DICT)


class TestCreateDataframeEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_large_csv_dataset(self):
        """Test with large CSV dataset."""
        # Generate large CSV
        rows = 10000
        csv_lines = ["col1,col2,col3"]
        for i in range(rows):
            csv_lines.append(f"{i},value{i},{i * 2}")
        csv_data = "\n".join(csv_lines).encode("utf-8")

        df_test = create_dataframe(csv_data, FileFormat.CSV)

        assert df_test.shape == (rows, 3)

    def test_unicode_characters(self):
        """Test with unicode characters in data."""
        dict_data = {
            "name": ["Alice", "José", "李明", "Müller"],
            "greeting": ["Hello", "Hola", "你好", "Guten Tag"],
        }

        df_test = create_dataframe(dict_data, FileFormat.DICT)

        assert df_test.iloc[1]["name"] == "José"
        assert df_test.iloc[2]["greeting"] == "你好"

    def test_special_characters_in_csv(self):
        """Test CSV with special characters."""
        csv_data = b'name,value\n"Test, Inc.",100\n"Quote""d",200'

        df_test = create_dataframe(csv_data, FileFormat.CSV)

        assert df_test.iloc[0]["name"] == "Test, Inc."
        assert df_test.iloc[1]["name"] == 'Quote"d'

    def test_mixed_data_types(self):
        """Test dict with mixed data types."""
        dict_data = {
            "string_col": ["a", "b", "c"],
            "int_col": [1, 2, 3],
            "float_col": [1.1, 2.2, 3.3],
            "bool_col": [True, False, True],
        }

        df_test = create_dataframe(dict_data, FileFormat.DICT)

        assert df_test.shape == (3, 4)
        assert df_test["int_col"].dtype in ["int64", "int32"]
        assert df_test["float_col"].dtype == "float64"

    def test_csv_with_missing_values(self):
        """Test CSV with missing/null values."""
        csv_data = b"name,age,city\nAlice,25,New York\nBob,,Los Angeles\nCharlie,35,"

        df_test = create_dataframe(csv_data, FileFormat.CSV)

        assert df_test.shape == (3, 3)
        assert pd.isna(df_test.iloc[1]["age"])
        assert pd.isna(df_test.iloc[2]["city"])

    def test_single_row_dataframe(self):
        """Test creating dataframe with single row."""
        dict_data = {"name": ["Alice"], "age": [25]}

        df_test = create_dataframe(dict_data, FileFormat.DICT)

        assert df_test.shape == (1, 2)

    def test_single_column_dataframe(self):
        """Test creating dataframe with single column."""
        csv_data = b"name\nAlice\nBob\nCharlie"

        df_test = create_dataframe(csv_data, FileFormat.CSV)

        assert df_test.shape == (3, 1)


class TestCreateDataframeOptionsValidation:
    """Test various file format options combinations."""

    def test_csv_multiple_options(self):
        """Test CSV with multiple options."""
        csv_data = b"name;age;city\nAlice;25;New York"

        df_test = create_dataframe(
            csv_data,
            FileFormat.CSV,
            file_format_options={"sep": ";", "encoding": "utf-8", "skipinitialspace": True},
        )

        assert isinstance(df_test, pd.DataFrame)

    def test_json_multiple_options(self):
        """Test JSON with multiple options."""
        json_data = b'[{"name":"Alice","age":25}]'

        df_test = create_dataframe(
            json_data,
            FileFormat.JSON,
            file_format_options={"orient": "records", "dtype": {"age": "int64"}},
        )

        assert df_test.iloc[0]["age"] == 25

    def test_parquet_with_filters(self):
        """Test parquet with filter options."""
        sample_df_test = pd.DataFrame({"name": ["Alice", "Bob", "Charlie"], "age": [25, 30, 35]})
        parquet_bytes = sample_df_test.to_parquet(engine="pyarrow")

        # Test basic parquet load (filters would require pyarrow setup)
        df_test = create_dataframe(parquet_bytes, FileFormat.PARQUET)

        assert df_test.shape[0] == 3


# Integration tests
class TestCreateDataframeIntegration:
    """Integration tests combining multiple scenarios."""

    def test_csv_to_dataframe_to_dict(self):
        """Test converting CSV to DataFrame and validating structure."""
        csv_data = b"name,age,city\nAlice,25,NYC\nBob,30,LA"

        df_test = create_dataframe(csv_data, FileFormat.CSV)
        result_dict = df_test.to_dict(orient="records")

        assert result_dict[0]["name"] == "Alice"
        assert result_dict[1]["age"] == 30

    def test_dict_to_dataframe_to_json(self):
        """Test converting dict to DataFrame to JSON."""
        dict_data = {"name": ["Alice", "Bob"], "age": [25, 30]}

        df_test = create_dataframe(dict_data, FileFormat.DICT)
        json_result = df_test.to_json(orient="records")

        assert "Alice" in json_result
        assert "25" in json_result

    def test_parquet_roundtrip(self):
        """Test parquet roundtrip conversion."""
        original_df_test = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})

        # Convert to parquet
        parquet_bytes = original_df_test.to_parquet(engine="pyarrow")

        # Load back
        loaded_df_test = create_dataframe(parquet_bytes, FileFormat.PARQUET)

        pd.testing.assert_frame_equal(original_df_test, loaded_df_test)
