# Binary Rain Helper Toolkit: Azure Cloud

`binaryrain_helper_cloud_azure` is a python package that aims to simplify and help with common functions in Azure Cloud areas. It builds on top of the `azure` library and provides additional functionality to make working with Azure Cloud easier, reduces boilerplate code and provides clear error messages.

## Key Functions

- `return_http_response()`: handles returning HTTP responses with status codes and messages:

  ```python
    from binaryrain_helper_cloud_azure import return_http_response
    import json

    # Return a 200 OK response
    return return_http_response('Success Message', 200)

    # Return json data with a 201 Created response
    return return_http_response(json.dumps({'key': 'value'}), 201)

    # Return a 404 Not Found response
    return return_http_response('Resource not found', 404)

    # Return a 500 Internal Server Error response
    return return_http_response('Internal Server Error', 500)
  ```

- `read_blob_data()`: provides a simplified way to read data from Azure Blob Storage:

  ```python
    from binaryrain_helper_cloud_azure import read_blob_data

    # Read a Parquet file from blob storage
    bytes = read_blob_data(
        blob_account="your_account",
        container_name="your_container",
        blob_name="data.parquet"
    )

    # Read CSV with custom format
    bytes = read_blob_data(
        blob_account="your_account",
        container_name="your_container",
        blob_name="data.csv",
    )
  ```

- `upload_blob_data()`: handles uploading dataframes to blob storage:

  ```python
      from binaryrain_helper_cloud_azure import upload_blob_data

      # Upload dataframe as Parquet
      upload_blob_data(
          blob_account="your_account",
          container_name="your_container",
          blob_name="data.parquet",
          df=your_dataframe
      )

      # Upload with compression options
      upload_blob_data(
          blob_account="your_account",
          container_name="your_container",
          blob_name="data.parquet",
          df=your_dataframe,
          file_format_options={'compression': 'snappy'}
      )
  ```
