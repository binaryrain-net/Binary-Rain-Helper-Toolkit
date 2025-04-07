# Binary Rain Helper Toolkit: AWS Cloud

`binaryrain_helper_cloud_aws` is a python package that aims to simplify and help with common functions in AWS Cloud areas. It builds on top of the `boto3` and `aws-lambda-powertools` libraries and provides additional functionality to make working with AWS Cloud easier, reduces boilerplate code and provides clear error messages.

## Key Functions

- `get_secret_data()`: retrieves secrets from AWS Secrets Manager:

  ```python
  from binaryrain_helper_cloud_aws.aws import get_secret_data

  # Get a secret from AWS Secrets Manager
  secret = get_secret_data("my-secret")

  # Access secret values
  database_password = secret["password"]
  ```

- `get_app_config()`: simplifies working with AWS AppConfig:

  ```python
  from binaryrain_helper_cloud_aws.aws import get_app_config

  # Load configuration from AWS AppConfig
  config = get_app_config(
      AppConfig_environment="Production",
      AppConfig_application="MyApp",
      AppConfig_profile="DefaultConfig"
  )

  # Access configuration values
  api_endpoint = config["api_endpoint"]
  ```

- `load_file_from_s3()`: provides a simple way to read data from S3:

  ```python
  from binaryrain_helper_cloud_aws.aws import load_file_from_s3

  # Load a file from S3
  file_bytes = load_file_from_s3(
      filename="data.csv",
      s3_bucket="my-bucket"
  )

  # Use the file content
  print(f"File size: {len(file_bytes)} bytes")
  ```

- `save_file_to_s3()`: handles uploading files to S3 with optional encryption:

  ```python
  from binaryrain_helper_cloud_aws.aws import save_file_to_s3

  # Save a file to S3
  save_file_to_s3(
      filename="output.json",
      s3_bucket="my-bucket",
      file_contents=json_bytes
  )

  # Save with server-side encryption
  save_file_to_s3(
      filename="sensitive-data.csv",
      s3_bucket="my-secure-bucket",
      file_contents=csv_bytes,
      server_side_encryption="aws:kms",
      sse_kms_key_id="arn:aws:kms:region:account:key/key-id"
  )
  ```

- `get_s3_presigned_url_readonly()`: generates presigned URLs for time-limited S3 object access:

  ```python
  from binaryrain_helper_cloud_aws.aws import get_s3_presigned_url_readonly

  # Generate a presigned URL valid for 2 minutes
  url = get_s3_presigned_url_readonly(
      filename="report.pdf",
      s3_bucket="my-bucket"
  )

  # Generate a presigned URL valid for 1 hour
  long_url = get_s3_presigned_url_readonly(
      filename="large-dataset.parquet",
      s3_bucket="my-bucket",
      expires_in=3600
  )
  ```

## Benefits

- Consistent error handling with clear messages
- Input validation for all function parameters
- Simplified authentication and access to AWS services
- Secure handling of secrets and sensitive information
- Type hints for better IDE support
