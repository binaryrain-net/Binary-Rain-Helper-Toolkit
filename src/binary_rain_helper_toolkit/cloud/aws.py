import json
import boto3
from aws_lambda_powertools import Logger
from aws_lambda_powertools.utilities import parameters
from botocore.exceptions import ClientError


# initialize the logger, logging will be handled via aws_lambda_powertools
logger = Logger()


def get_secret_data(secret_name: str) -> dict:
    """
    Get secret data from AWS Secrets Manager.
    Returns
    -------
    secrets_data : dict
        dictionary with secret data.
    """
    if not secret_name:
        raise ValueError("No secret name provided.")

    try:
        secret_data = parameters.get_secret(secret_name, transform="json")
    except parameters.exceptions.GetParameterError as exc:
        msg = f"Error getting secret for {secret_name} from SSM. See logs for more details."
        logger.exception(msg + f" Exception: {exc}")
        raise ValueError(msg) from exc
    except parameters.exceptions.TransformParameterError as exc:
        msg = f"Error transforming the secret for {secret_name}. See logs for more details."
        logger.exception(msg + f" Exception: {exc}")
        raise ValueError(msg) from exc

    return secret_data


def get_app_config(AppConfig_environment: str, AppConfig_application: str, AppConfig_profile: str) -> dict:
    """
    Load confiugration from AWS AppConfig.

    Parameters
    ----------
    AppConfig_environment : str
        Environment name
    AppConfig_application : str
        Application name
    AppConfig_profile : str
        Profile name

    Returns
    -------
    app_config : dict
        Configuration data for the app.
    """
    
    # validate input parameters
    if not AppConfig_environment:
        raise ValueError("No environment provided.")
    if not AppConfig_application:
        raise ValueError("No application provided.")
    if not AppConfig_profile:
        raise ValueError("No profile provided.")

    try:
        app_config: bytes = parameters.get_app_config(
            name = AppConfig_profile,
            environment = AppConfig_environment,
            application = AppConfig_application
        )
    except ClientError as exc:
        msg = "Error loading configuration data from AppConfig. See logs for more details."
        logger.exception(msg + f" Exception: {exc}")
        raise ValueError(msg) from exc
    
    # load the configuration data
    try:
        app_config = json.loads(app_config)
    except json.JSONDecodeError as exc:
        msg = "Error decoding configuration data from AppConfig. See logs for more details."
        logger.exception(msg + f" Exception: {exc}")
        raise ValueError(msg) from exc

    return app_config


def load_file_from_s3(filename: str, s3_bucket: str) -> bytes:
    """
    Load file from S3 bucket.

    ### Parameters
    ----------
    filename : str
        name of the file to load.
    s3_bucket : str
        name of the S3 bucket where the file is stored.

    ### Returns
    -------
    fileobj : bytes
        file object as bytes.
    exception : Exception
        exception if the file cannot be loaded from S3.
    """

    # validate input parameters
    if not filename:
        raise ValueError("No filename provided.")
    if not s3_bucket:
        raise ValueError("No S3 bucket provided.")


    try:
        s3_client = boto3.client("s3")
        file_obj = s3_client.get_object(
            Bucket = s3_bucket,
            Key = filename
        )
    except ClientError as exc:
        error_msg = f"Could not load file {filename} from S3. Check logs for more details."
        logger.exception(error_msg + f" Exception: {exc}")
        raise ValueError(error_msg) from exc

    return file_obj["Body"].read()


def save_file_to_s3(filename: str, s3_bucket: str, file_contents: bytes, server_side_encryption: str = None, sse_kms_key_id: str = None) -> bool:
    """
    Save file to S3 bucket.

    ### Parameters
    ----------
    filename : str
        name of the file in S3 for which the presigned URL is generated.
            if not provided, an exception is raised.
    s3_bucket : str
        name of the S3 bucket where the file is stored.
            if not provided, an exception is raised.
    file_contents : bytes
        file contents as bytes.
            if not provided, an exception is raised.
    ssekms_key_id : str
        KMS key ID for server side encryption.
            default is None.
    server_side_encryption : str
        server side encryption type.
            default is None.

    ### Returns
    -------
    True : bool
        True if the file was saved successfully.
    exception : Exception
        exception if the file cannot be saved to S3.
    """

    # validate input parameters
    if not filename:
        raise ValueError("No filename provided.")
    if not s3_bucket:
        raise ValueError("No S3 bucket provided.")
    if not file_contents or not isinstance(file_contents, bytes) or len(file_contents) == 0:
        raise ValueError("No file contents provided or file contents are empty or not of type bytes.")

    # if server side encryption is provided, make sure the KMS key ID is also provided
    if server_side_encryption and not sse_kms_key_id:
        raise ValueError("SSE requested, but no KMS key ID provided for server side encryption.")
    
    try:
        s3_client = boto3.client("s3")

        # if server side encryption is provided, use it
        if server_side_encryption:
            s3_client.put_object(
                Bucket = s3_bucket,
                Key = filename,
                Body = file_contents,
                ServerSideEncryption = server_side_encryption,
                SSEKMSKeyId = sse_kms_key_id,
            )
        else:
            s3_client.put_object(
                Bucket = s3_bucket,
                Key = filename,
                Body = file_contents,
            )
    except ClientError as exc:
        error_msg = f"Error saving {filename} to S3."
        logger.exception(error_msg + f" Exception: {exc}")
        raise ValueError(error_msg) from exc
    return True


def get_s3_presigned_url_readonly(filename: str, s3_bucket: str, expires_in: int = 120) -> str:
    """
    Get a presigned URL for a file in S3.

    ### Parameters
    ----------
    filename : str
        name of the file in S3 for which the presigned URL is generated.
            if not provided, an exception is raised.
    s3_bucket : str
        name of the S3 bucket where the file is stored.
            if not provided, an exception is raised.
    expires_in : int
        time in seconds for which the presigned URL is valid.
            default is 120 seconds.

    ### Returns
    -------
    presigned_url : str
        presigned URL for the file in S3.
    exception : Exception
        exception if the presigned URL cannot be created.
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
            ClientMethod = "get_object",
            Params = {
                "Bucket": s3_bucket,
                "Key": filename
            },
            ExpiresIn = expires_in
        )
    except ClientError as exc:
        error_msg = "Could not create presigned URL. Check logs for more details."
        logger.exception(error_msg + f" Exception: {exc}")
        raise ValueError(error_msg) from exc

    # return the presigned URL
    return presigned_url
