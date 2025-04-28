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

- `combine_dataframes()`: provides a simple way to combine multiple DataFrames:

  ```python
    from binaryrain_helper_data_processing import combine_dataframes

    # ....df1 and df2 are pandas DataFrames

    # Combine DataFrames
    combined_df = combine_dataframes(df1, df2, sort=True)
  ```

- `convert_todatetime()`: automatically detects and converts date columns:

  Supports common date formats:

  - %d.%m.%Y (e.g., "31.12.2023")
  - %Y-%m-%d (e.g., "2023-12-31")
  - %Y-%m-%d %H:%M:%S (e.g., "2023-12-31 23:59:59")
  - %Y-%m-%dT%H:%M:%S (ISO format)

  ```python
      from binaryrain_helper_data_processing import convert_todatetime

      # ....df is a pandas DataFrame

      # Convert date columns
      df = convert_todatetime(df)
  ```

- `format_datetime_columns()`: formats specific datetime columns:

  ```python
      from binaryrain_helper_data_processing import format_datetime_columns

      # ....df is a pandas DataFrame

      # Format date columns directly
      df = format_datetime_columns(df, datetime_columns=['date_column1', 'date_column2'], datetime_format='%Y-%m-%d')

      # Format date columns to in string columns
      df = format_datetime_columns(df, datetime_columns=['date_column1', 'date_column2'], datetime_format='%Y-%m-%d', datetime_columns=['string_column1', 'string_column2'])
  ```

- `clean_dataframe()`: cleans DataFrames by removing duplicates and missing values:

  ```python
      from binaryrain_helper_data_processing import clean_dataframe

      # ....df is a pandas DataFrame

      # Clean DataFrame
      df = clean_dataframe(df)
  ```

- `remove_empty_values()`: filters specific columns:

  ```python
      from binaryrain_helper_data_processing import remove_empty_values

      # ....df is a pandas DataFrame

      # Remove empty values
      df = remove_empty_values(df, filter_column'column1')
  ```

- `format_numeric_values()`: handles locale-specific number formatting:

  ```python
      from binaryrain_helper_data_processing import format_numeric_values

      # ....df is a pandas DataFrame

      # Convert European number format (1.234,56) to standard format (1,234.56)
      df = format_numeric_values(
            df,
            columns=['price', 'quantity'],
            swap_separators=True,
            old_decimal_separator=',',
            old_thousands_separator='.',
            decimal_separator='.',
            thousands_separator=',',
        )
  ```

## Benefits

- Consistent interface for different file formats
- Simplified error handling with clear messages
- Optional format-specific configurations
- Built on pandas for robust data processing
- Type hints for better IDE support
