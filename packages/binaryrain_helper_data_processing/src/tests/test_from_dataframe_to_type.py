"""
Comprehensive test suite for the from_dataframe_to_type function.
Tests all file formats, options, edge cases, and error handling.
"""

from io import StringIO
import pytest
import pandas as pd
from binaryrain_helper_data_processing.dataframe import from_dataframe_to_type, FileFormat


class TestFromDataframeToTypeCSV:
    """Test cases for CSV file format conversion."""

    def test_csv_basic(self):
        """Test basic CSV conversion without options."""
        df_test = pd.DataFrame(
            {
                "name": ["Alice", "Bob", "Charlie"],
                "age": [25, 30, 35],
                "city": ["New York", "Los Angeles", "Chicago"],
            }
        )

        csv_content = from_dataframe_to_type(df_test, FileFormat.CSV)

        assert isinstance(csv_content, str)
        assert "name,age,city" in csv_content
        assert "Alice,25,New York" in csv_content
        assert "Bob,30,Los Angeles" in csv_content
        assert "Charlie,35,Chicago" in csv_content

    def test_csv_with_options(self):
        """Test CSV conversion with custom options (e.g., different delimiter)."""
        df_test = pd.DataFrame(
            {"name": ["Alice", "Bob"], "age": [25, 30], "city": ["New York", "Los Angeles"]}
        )

        csv_content = from_dataframe_to_type(
            df_test, FileFormat.CSV, file_format_options={"sep": ";"}
        )

        assert isinstance(csv_content, str)
        assert "name;age;city" in csv_content
        assert "Alice;25;New York" in csv_content

    def test_csv_with_encoding(self):
        """Test CSV conversion with encoding options."""
        df_test = pd.DataFrame({"name": ["Test", "Data"], "value": [100, 200]})

        csv_content = from_dataframe_to_type(
            df_test, FileFormat.CSV, file_format_options={"encoding": "utf-8"}
        )

        assert isinstance(csv_content, str)
        assert "Test,100" in csv_content

    def test_csv_empty_dataframe(self):
        """Test CSV conversion with empty dataframe."""
        df_test = pd.DataFrame()

        csv_content = from_dataframe_to_type(df_test, FileFormat.CSV)

        assert isinstance(csv_content, str)


class TestFromDataframeToTypeDICT:
    """Test cases for DICT file format conversion."""

    def test_dict_basic(self):
        """Test basic dict conversion without options."""
        df_test = pd.DataFrame(
            {
                "name": ["Alice", "Bob", "Charlie"],
                "age": [25, 30, 35],
                "city": ["New York", "Los Angeles", "Chicago"],
            }
        )

        dict_content = from_dataframe_to_type(df_test, FileFormat.DICT)

        assert isinstance(dict_content, list)
        assert len(dict_content) == 3
        assert dict_content[0] == {"name": "Alice", "age": 25, "city": "New York"}
        assert dict_content[1] == {"name": "Bob", "age": 30, "city": "Los Angeles"}
        assert dict_content[2] == {"name": "Charlie", "age": 35, "city": "Chicago"}

    def test_dict_with_orient_option(self):
        """Test dict conversion with orient option."""
        df_test = pd.DataFrame({"name": ["Alice", "Bob"], "age": [25, 30]})

        dict_content = from_dataframe_to_type(
            df_test, FileFormat.DICT, file_format_options={"orient": "list"}
        )

        assert isinstance(dict_content, dict)
        assert dict_content["name"] == ["Alice", "Bob"]
        assert dict_content["age"] == [25, 30]

    def test_dict_with_columns_orient(self):
        """Test dict conversion with columns orient."""
        df_test = pd.DataFrame({"name": ["Alice", "Bob"], "age": [25, 30]})

        dict_content = from_dataframe_to_type(
            df_test, FileFormat.DICT, file_format_options={"orient": "dict"}
        )

        assert isinstance(dict_content, dict)
        assert "name" in dict_content
        assert "age" in dict_content

    def test_dict_empty_dataframe(self):
        """Test dict conversion with empty dataframe."""
        df_test = pd.DataFrame()

        dict_content = from_dataframe_to_type(df_test, FileFormat.DICT)

        assert isinstance(dict_content, list)
        assert len(dict_content) == 0

    def test_dict_single_column(self):
        """Test dict conversion with single column."""
        df_test = pd.DataFrame({"name": ["Alice", "Bob"]})

        dict_content = from_dataframe_to_type(df_test, FileFormat.DICT)

        assert isinstance(dict_content, list)
        assert len(dict_content) == 2
        assert dict_content[0] == {"name": "Alice"}


class TestFromDataframeToTypePARQUET:
    """Test cases for PARQUET file format conversion."""

    def test_parquet_basic(self):
        """Test basic parquet conversion without options."""
        df_test = pd.DataFrame(
            {
                "name": ["Alice", "Bob", "Charlie"],
                "age": [25, 30, 35],
                "city": ["New York", "Los Angeles", "Chicago"],
            }
        )

        parquet_bytes = from_dataframe_to_type(df_test, FileFormat.PARQUET)

        assert isinstance(parquet_bytes, bytes)
        # Verify by reading back
        df_result = pd.read_parquet(pd.io.common.BytesIO(parquet_bytes), engine="pyarrow")
        pd.testing.assert_frame_equal(df_test, df_result)

    def test_parquet_with_compression(self):
        """Test parquet conversion with compression option."""
        df_test = pd.DataFrame({"name": ["Alice", "Bob"], "age": [25, 30]})

        parquet_bytes = from_dataframe_to_type(
            df_test, FileFormat.PARQUET, file_format_options={"compression": "gzip"}
        )

        assert isinstance(parquet_bytes, bytes)
        # Verify by reading back
        df_result = pd.read_parquet(pd.io.common.BytesIO(parquet_bytes), engine="pyarrow")
        pd.testing.assert_frame_equal(df_test, df_result)

    def test_parquet_empty_dataframe(self):
        """Test parquet conversion with empty dataframe."""
        df_test = pd.DataFrame()

        parquet_bytes = from_dataframe_to_type(df_test, FileFormat.PARQUET)

        assert isinstance(parquet_bytes, bytes)

    def test_parquet_with_mixed_types(self):
        """Test parquet conversion with mixed data types."""
        df_test = pd.DataFrame(
            {
                "string_col": ["a", "b", "c"],
                "int_col": [1, 2, 3],
                "float_col": [1.1, 2.2, 3.3],
                "bool_col": [True, False, True],
            }
        )

        parquet_bytes = from_dataframe_to_type(df_test, FileFormat.PARQUET)

        assert isinstance(parquet_bytes, bytes)
        # Verify by reading back
        df_result = pd.read_parquet(pd.io.common.BytesIO(parquet_bytes), engine="pyarrow")
        pd.testing.assert_frame_equal(df_test, df_result)

    def test_parquet_single_row(self):
        """Test parquet conversion with single row."""
        df_test = pd.DataFrame({"name": ["Alice"], "age": [25]})

        parquet_bytes = from_dataframe_to_type(df_test, FileFormat.PARQUET)

        assert isinstance(parquet_bytes, bytes)


class TestFromDataframeToTypeJSON:
    """Test cases for JSON file format conversion."""

    def test_json_basic(self):
        """Test basic JSON conversion without options."""
        df_test = pd.DataFrame(
            {"name": ["Alice", "Bob"], "age": [25, 30], "city": ["New York", "Los Angeles"]}
        )

        json_content = from_dataframe_to_type(df_test, FileFormat.JSON)

        assert isinstance(json_content, str)
        assert "Alice" in json_content
        assert "25" in json_content
        assert "Bob" in json_content

    def test_json_with_orient_records(self):
        """Test JSON conversion with records orient."""
        df_test = pd.DataFrame({"name": ["Alice", "Bob"], "age": [25, 30]})

        json_content = from_dataframe_to_type(
            df_test, FileFormat.JSON, file_format_options={"orient": "records"}
        )

        assert isinstance(json_content, str)
        assert "Alice" in json_content
        # Should be array format
        assert "[" in json_content

    def test_json_with_orient_columns(self):
        """Test JSON conversion with columns orient."""
        df_test = pd.DataFrame({"name": ["Alice", "Bob"], "age": [25, 30]})

        json_content = from_dataframe_to_type(
            df_test, FileFormat.JSON, file_format_options={"orient": "columns"}
        )

        assert isinstance(json_content, str)
        assert "name" in json_content
        assert "age" in json_content

    def test_json_empty_dataframe(self):
        """Test JSON conversion with empty dataframe."""
        df_test = pd.DataFrame()

        json_content = from_dataframe_to_type(df_test, FileFormat.JSON)

        assert isinstance(json_content, str)

    def test_json_with_indent(self):
        """Test JSON conversion with indent option."""
        df_test = pd.DataFrame({"name": ["Alice"], "age": [25]})

        json_content = from_dataframe_to_type(
            df_test, FileFormat.JSON, file_format_options={"indent": 2}
        )

        assert isinstance(json_content, str)
        # Indented JSON should have newlines
        assert "\n" in json_content or json_content.count(" ") > 10

    def test_json_with_date_format(self):
        """Test JSON conversion with date format option."""
        df_test = pd.DataFrame(
            {"name": ["Alice"], "date": [pd.Timestamp("2023-01-01")], "value": [100]}
        )

        json_content = from_dataframe_to_type(
            df_test, FileFormat.JSON, file_format_options={"date_format": "iso"}
        )

        assert isinstance(json_content, str)
        assert "2023" in json_content


class TestFromDataframeToTypeErrorHandling:
    """Test cases for error handling and edge cases."""

    def test_unknown_file_format(self):
        """Test with unknown file format enum value."""
        df_test = pd.DataFrame({"name": ["Alice"], "age": [25]})

        # Create a mock unknown format
        class UnknownFormat:
            pass

        with pytest.raises(ValueError, match="Error converting dataframe"):
            from_dataframe_to_type(df_test, UnknownFormat)

    def test_none_dataframe(self):
        """Test with None as dataframe."""
        with pytest.raises(ValueError, match="Error converting dataframe"):
            from_dataframe_to_type(None, FileFormat.CSV)

    def test_invalid_dataframe_type(self):
        """Test with invalid dataframe type."""
        with pytest.raises(ValueError, match="Error converting dataframe"):
            from_dataframe_to_type("not a dataframe", FileFormat.CSV)

    def test_invalid_options(self):
        """Test with invalid format options."""
        df_test = pd.DataFrame({"name": ["Alice"], "age": [25]})

        # Test with invalid option that pandas won't accept
        with pytest.raises(ValueError, match="Error converting dataframe"):
            from_dataframe_to_type(
                df_test, FileFormat.CSV, file_format_options={"invalid_param": True}
            )

    def test_dict_type(self):
        """Test passing dict instead of DataFrame."""
        dict_data = {"name": ["Alice"], "age": [25]}

        with pytest.raises(ValueError, match="Error converting dataframe"):
            from_dataframe_to_type(dict_data, FileFormat.CSV)


class TestFromDataframeToTypeEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_large_dataframe(self):
        """Test with large dataframe."""
        rows = 10000
        df_test = pd.DataFrame(
            {"col1": range(rows), "col2": [f"value{i}" for i in range(rows)], "col3": range(rows)}
        )

        csv_content = from_dataframe_to_type(df_test, FileFormat.CSV)

        assert isinstance(csv_content, str)
        assert len(csv_content) > 0

    def test_unicode_characters(self):
        """Test with unicode characters in data."""
        df_test = pd.DataFrame(
            {
                "name": ["Alice", "José", "李明", "Müller"],
                "greeting": ["Hello", "Hola", "你好", "Guten Tag"],
            }
        )

        csv_content = from_dataframe_to_type(df_test, FileFormat.CSV)

        assert isinstance(csv_content, str)
        assert "José" in csv_content
        assert "李明" in csv_content
        assert "你好" in csv_content

    def test_special_characters_csv(self):
        """Test CSV with special characters."""
        df_test = pd.DataFrame({"name": ["Test, Inc.", 'Quote"d'], "value": [100, 200]})

        csv_content = from_dataframe_to_type(df_test, FileFormat.CSV)

        assert isinstance(csv_content, str)
        # CSV should properly escape special characters

    def test_null_values(self):
        """Test with null/NaN values."""
        df_test = pd.DataFrame(
            {
                "name": ["Alice", "Bob", "Charlie"],
                "age": [25, None, 35],
                "city": ["NYC", "LA", None],
            }
        )

        csv_content = from_dataframe_to_type(df_test, FileFormat.CSV)

        assert isinstance(csv_content, str)
        assert "Alice" in csv_content

    def test_single_row_dataframe(self):
        """Test converting single row dataframe."""
        df_test = pd.DataFrame({"name": ["Alice"], "age": [25]})

        dict_content = from_dataframe_to_type(df_test, FileFormat.DICT)

        assert isinstance(dict_content, list)
        assert len(dict_content) == 1
        assert dict_content[0] == {"name": "Alice", "age": 25}

    def test_single_column_dataframe(self):
        """Test converting single column dataframe."""
        df_test = pd.DataFrame({"name": ["Alice", "Bob", "Charlie"]})

        csv_content = from_dataframe_to_type(df_test, FileFormat.CSV)

        assert isinstance(csv_content, str)
        assert "name" in csv_content
        assert "Alice" in csv_content

    def test_dataframe_with_multi_index(self):
        """Test dataframe with multi-index."""
        df_test = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        df_test.index = pd.MultiIndex.from_tuples([("a", 1), ("a", 2), ("b", 1)])

        dict_content = from_dataframe_to_type(df_test, FileFormat.DICT)

        assert isinstance(dict_content, list)


class TestFromDataframeToTypeOptionsValidation:
    """Test various file format options combinations."""

    def test_csv_multiple_options(self):
        """Test CSV with multiple options."""
        df_test = pd.DataFrame({"name": ["Alice"], "age": [25], "city": ["New York"]})

        csv_content = from_dataframe_to_type(
            df_test,
            FileFormat.CSV,
            file_format_options={"sep": ";", "encoding": "utf-8", "lineterminator": "\n"},
        )

        assert isinstance(csv_content, str)
        assert ";" in csv_content

    def test_json_multiple_options(self):
        """Test JSON with multiple options."""
        df_test = pd.DataFrame({"name": ["Alice"], "age": [25]})

        json_content = from_dataframe_to_type(
            df_test,
            FileFormat.JSON,
            file_format_options={"orient": "records", "indent": 2},
        )

        assert isinstance(json_content, str)
        assert "Alice" in json_content

    def test_parquet_compression_options(self):
        """Test parquet with compression options."""
        df_test = pd.DataFrame({"name": ["Alice", "Bob", "Charlie"], "age": [25, 30, 35]})

        parquet_bytes = from_dataframe_to_type(
            df_test, FileFormat.PARQUET, file_format_options={"compression": "snappy"}
        )

        assert isinstance(parquet_bytes, bytes)

    def test_dict_multiple_orients(self):
        """Test dict with different orient options."""
        df_test = pd.DataFrame({"name": ["Alice", "Bob"], "age": [25, 30]})

        # Test 'split' orient
        dict_content = from_dataframe_to_type(
            df_test, FileFormat.DICT, file_format_options={"orient": "split"}
        )

        assert isinstance(dict_content, dict)
        assert "columns" in dict_content
        assert "data" in dict_content


# Integration tests
class TestFromDataframeToTypeIntegration:
    """Integration tests combining multiple scenarios."""

    def test_csv_roundtrip(self):
        """Test CSV roundtrip conversion."""
        original_df_test = pd.DataFrame(
            {"name": ["Alice", "Bob"], "age": [25, 30], "city": ["NYC", "LA"]}
        )

        # Convert to CSV
        csv_content = from_dataframe_to_type(original_df_test, FileFormat.CSV)
        # Convert back to DataFrame
        loaded_df_test = pd.read_csv(StringIO(csv_content))

        pd.testing.assert_frame_equal(original_df_test, loaded_df_test)

    def test_dict_roundtrip(self):
        """Test dict roundtrip conversion."""
        original_df_test = pd.DataFrame({"name": ["Alice", "Bob"], "age": [25, 30]})

        # Convert to dict
        dict_content = from_dataframe_to_type(original_df_test, FileFormat.DICT)

        # Convert back to DataFrame
        loaded_df_test = pd.DataFrame(dict_content)

        pd.testing.assert_frame_equal(original_df_test, loaded_df_test)

    def test_parquet_roundtrip(self):
        """Test parquet roundtrip conversion."""
        original_df_test = pd.DataFrame(
            {"col1": [1, 2, 3], "col2": ["a", "b", "c"], "col3": [1.1, 2.2, 3.3]}
        )

        # Convert to parquet
        parquet_bytes = from_dataframe_to_type(original_df_test, FileFormat.PARQUET)

        # Convert back to DataFrame
        loaded_df_test = pd.read_parquet(pd.io.common.BytesIO(parquet_bytes), engine="pyarrow")

        pd.testing.assert_frame_equal(original_df_test, loaded_df_test)

    def test_json_roundtrip(self):
        """Test JSON roundtrip conversion."""
        original_df_test = pd.DataFrame({"name": ["Alice", "Bob"], "age": [25, 30]})

        # Convert to JSON
        json_content = from_dataframe_to_type(
            original_df_test, FileFormat.JSON, file_format_options={"orient": "records"}
        )

        # Convert back to DataFrame
        loaded_df_test = pd.read_json(StringIO(json_content), orient="records")

        pd.testing.assert_frame_equal(original_df_test, loaded_df_test)

    def test_multiple_format_conversions(self):
        """Test converting through multiple formats."""
        original_df_test = pd.DataFrame({"name": ["Alice", "Bob"], "value": [100, 200]})

        # CSV -> Dict -> JSON
        csv_content = from_dataframe_to_type(original_df_test, FileFormat.CSV)
        dict_content = from_dataframe_to_type(original_df_test, FileFormat.DICT)
        json_content = from_dataframe_to_type(original_df_test, FileFormat.JSON)

        assert isinstance(csv_content, str)
        assert isinstance(dict_content, list)
        assert isinstance(json_content, str)
