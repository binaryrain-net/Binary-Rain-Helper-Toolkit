import boto3
from aws_lambda_powertools.utilities import parameters
from botocore.exceptions import ClientError


def get_secret_data(secret_name: str) -> dict:
    """
    Get secret data from AWS Secrets Manager.

    :param str secret_name:
        Name of the secret to retrieve.

    :returns dict:
        Secret data as a dictionary.
    :raises ValueError:
        If the secret name is not provided or if there is an error retrieving the secret.
    """
    if not secret_name:
        raise ValueError("No secret name provided.")

    try:
        secret_data = parameters.get_secret(secret_name, transform="json")
    except parameters.exceptions.GetParameterError as exc:
        raise ValueError(
            f"Error getting secret for {secret_name} from SSM. Exception: {exc}"
        ) from exc
    except parameters.exceptions.TransformParameterError as exc:
        raise ValueError(
            f"Error transforming the secret for {secret_name}. Exception: {exc}"
        ) from exc

    return secret_data


def get_app_config(
    AppConfig_environment: str, AppConfig_application: str, AppConfig_profile: str
) -> dict:
    """
    Load configuration from AWS AppConfig.

    :param str AppConfig_environment:
        Name of the AppConfig environment.
    :param str AppConfig_application:
        Name of the AppConfig application.
    :param str AppConfig_profile:
        Name of the AppConfig profile.

    :returns dict:
        Configuration data as a dictionary.
    :raises ValueError:
        If any of the parameters are not provided or if there is an error retrieving the configuration.
    """

    # validate input parameters
    if not AppConfig_environment:
        raise ValueError("No environment provided.")
    if not AppConfig_application:
        raise ValueError("No application provided.")
    if not AppConfig_profile:
        raise ValueError("No profile provided.")

    try:
        app_config = parameters.get_app_config(
            name=AppConfig_profile,
            environment=AppConfig_environment,
            application=AppConfig_application,
            transform="json",
        )
    except ClientError as exc:
        raise ValueError(
            f"Error loading configuration data from AppConfig. Exception: {exc}"
        ) from exc

    return app_config


def load_file_from_s3(filename: str, s3_bucket: str) -> bytes:
    """
    Load file from S3 bucket.

    :param str filename:
        Name of the file in S3 to load.
    :param str s3_bucket:
        Name of the S3 bucket where the file is stored.

    :returns bytes:
        File contents as bytes.
    :raises ValueError:
        If the filename or S3 bucket is not provided or if there is an error loading the file.
    """

    # validate input parameters
    if not filename:
        raise ValueError("No filename provided.")
    if not s3_bucket:
        raise ValueError("No S3 bucket provided.")

    try:
        s3_client = boto3.client("s3")
        file_obj = s3_client.get_object(Bucket=s3_bucket, Key=filename)
    except ClientError as exc:
        raise ValueError(
            f"Could not load file {filename} from S3. Exception: {exc}"
        ) from exc

    return file_obj["Body"].read()


def save_file_to_s3(
    filename: str,
    s3_bucket: str,
    file_contents: bytes,
    server_side_encryption: str = None,
    sse_kms_key_id: str = None,
) -> bool:
    """
    Save file to S3 bucket.

    :param str filename:
        Name of the file to save in S3.
    :param str s3_bucket:
        Name of the S3 bucket where the file will be saved.
    :param bytes file_contents:
        Contents of the file to save.
    :param str server_side_encryption: (optional)
        Type of server side encryption.
    :param str sse_kms_key_id: (optional)
        KMS key ID for server side encryption.

    :returns bool:
        Indicates whether the file got saved successfully, otherwise false.
    :raises ValueError:
        If the filename, S3 bucket, or file contents are not provided or if there is an error saving the file.
    """

    # validate input parameters
    if not filename:
        raise ValueError("No filename provided.")
    if not s3_bucket:
        raise ValueError("No S3 bucket provided.")
    if (
        not file_contents
        or not isinstance(file_contents, bytes)
        or len(file_contents) == 0
    ):
        raise ValueError(
            "No file contents provided or file contents are empty or not of type bytes."
        )

    # if server side encryption is provided, make sure the KMS key ID is also provided
    if server_side_encryption and not sse_kms_key_id:
        raise ValueError(
            "SSE requested, but no KMS key ID provided for server side encryption."
        )

    try:
        s3_client = boto3.client("s3")

        # if server side encryption is provided, use it
        if server_side_encryption:
            s3_client.put_object(
                Bucket=s3_bucket,
                Key=filename,
                Body=file_contents,
                ServerSideEncryption=server_side_encryption,
                SSEKMSKeyId=sse_kms_key_id,
            )
        else:
            s3_client.put_object(
                Bucket=s3_bucket,
                Key=filename,
                Body=file_contents,
            )
    except ClientError as exc:
        raise ValueError(f"Error saving {filename} to S3. Exception: {exc}") from exc
    return True


def get_s3_presigned_url_readonly(
    filename: str, s3_bucket: str, expires_in: int = 120
) -> str:
    """
    Get a presigned URL for a file in S3.

    :param str filename:
        Name of the file in S3.
    :param str s3_bucket:
        Name of the S3 bucket where the file is stored.
    :param int expires_in: (optional)
        Expiration time for the presigned URL in seconds. Default is 120 seconds.

    :returns str:
        Presigned URL for the file in S3.
    :raises ValueError:
        If the filename or S3 bucket is not provided or if there is an error generating the presigned URL.
    """

    # validate input parameters
    if not filename:
        raise ValueError("No filename provided.")
    if not s3_bucket:
        raise ValueError("No S3 bucket provided.")

    try:
        # create S3 client
        s3_client = boto3.client("s3")
        presigned_url = s3_client.generate_presigned_url(
            ClientMethod="get_object",
            Params={"Bucket": s3_bucket, "Key": filename},
            ExpiresIn=expires_in,
        )
    except ClientError as exc:
        raise ValueError(
            f"Could not create presigned URL. Check logs for more details. Exception: {exc}"
        ) from exc

    # return the presigned URL
    return presigned_url
