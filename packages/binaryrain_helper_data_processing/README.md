# Binary Rain Helper Toolkit: Data Processing

`binaryrain_helper_data_processing` is a python package that aims to simplify and help with common functions data processing areas. It builds on top of the `pandas` library and provides additional functionality to make data processing easier, reduces boilerplate code and provides clear error messages.

## Supported File Formats

- `PARQUET`: For efficient columnar storage
- `CSV`: For common tabular data
- `JSON`: For structured data exchange
- `DICT`: For Python dictionary data

## Key Functions

- `create_dataframe()` simplifies creating pandas DataFrames from various formats:

  ```python
    from binaryrain_helper_data_processing import FileFormat, create_dataframe

    # Create from CSV bytes
    df = create_dataframe(csv_bytes, FileFormat.CSV)

    # Create with custom options
    df = create_dataframe(parquet_bytes, FileFormat.PARQUET,
    file_format_options={'engine': 'pyarrow'})
  ```

- `convert_dataframe_to_type()`: handles converting DataFrames to different formats:

  ```python
    from binaryrain_helper_data_processing import FileFormat, convert_dataframe_to_type

    # ....df is a pandas DataFrame

    # Convert to CSV bytes
    csv_bytes = convert_dataframe_to_type(df, FileFormat.CSV)

    # Convert with custom options
    parquet_bytes = convert_dataframe_to_type(df, FileFormat.PARQUET,
    file_format_options={'engine': 'pyarrow'})
  ```

- `merge_dataframes()`: provides a simple way to merge multiple DataFrames:

  ```python
    from binaryrain_helper_data_processing import merge_dataframes

    # ....df1 and df2 are pandas DataFrames

    # Merge DataFrames
    merged_df = merge_dataframes(df1, df2, sort=True)
  ```

## Benefits
- Consistent interface for different file formats
- Simplified error handling with clear messages
- Optional format-specific configurations
- Built on pandas for robust data processing
- Type hints for better IDE support
