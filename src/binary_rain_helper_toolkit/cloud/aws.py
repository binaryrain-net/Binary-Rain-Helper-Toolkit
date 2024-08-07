import json
import boto3
from aws_lambda_powertools import Logger
from aws_lambda_powertools.utilities import parameters
from botocore.exceptions import ClientError


# initialize the logger, logging will be handled via aws_lambda_powertools
logger = Logger()


def get_logger() -> Logger:
    """
    Initializing the aws logger, logging will be handled via aws_lambda_powertools
    Returns
    -------
    logger : Logger
        logger object.
    """
    return Logger()


def get_secrets_data(secret_name: str, logger: Logger = logger) -> dict:
    """
    Get secrets data from AWS Secrets Manager.
    Returns
    -------
    secrets_data : dict
        dictionary with secrets data.
    """
    if not secret_name:
        raise ValueError("No secret name provided.")

    try:
        credentials = parameters.get_secret(secret_name, transform="json")
    except parameters.exceptions.GetParameterError as exc:
        logger.exception(
            f"Error getting credentials for {secret_name} from SSM, exception: {exc}"
        )
        raise ValueError("Error getting credentials from SSM") from exc
    except parameters.exceptions.TransformParameterError as exc:
        logger.exception(
            f"Error transofrming the credentials for {secret_name}, exception: {exc}"
        )
        raise ValueError("Error transofrming the credentials") from exc
    return credentials


def get_app_confiugration(
    app_config_name: str,
    app_config_env: str,
    app_config_app: str,
    data_handler: str | None = None,
    logger: Logger = logger,
) -> dict:
    """
    Load confiugration from AWS AppConfig.

    ### Parameters
    ----------
    app_config_name : str
        name of the AppConfig.
    app_config_env : str
        name of the AppConfig environment.
    app_config_app : str
        name of the AppConfig application.
    data_handler : str, optional
        data handler - internal name of the data handler.
            used for directory structure in S3. Equivalent to a directory
    logger : Logger
        logger object.
            default is the logger object from the module.

    ### Returns
    -------
    app_config : dict
        dictionary with the configuration data.
    exception : Exception
        exception if the configuration data cannot be loaded.
    """
    try:
        app_config: bytes = parameters.get_app_config(
            name=app_config_name, environment=app_config_env, application=app_config_app
        )
        app_config = json.loads(app_config.decode("utf-8"))
    except ClientError as exc:
        error_msg = (
            f"Error loading servey configuration data from AppConfig, exception: {exc}"
        )
        logger.error(error_msg)
        raise ClientError(error_msg) from exc

    if data_handler:
        # return only the handler provided
        return app_config[data_handler][0]
    else:
        # if no specific handler is provided, return all handlers
        return app_config


def validate_dataset_existance(
    data_handler: str,
    dataset: str,
    app_config_name: str,
    app_config_env: str,
    app_config_app: str,
    logger: Logger = logger,
):
    """
    Validate if the dataset is in the list of allowed datasets.

    ### Parameters
    ----------
    data_handler : str
        data handler - internal name of the data handler.
            used for directory structure in S3. Equivalent to a directory
    dataset : str
        dataset - internal name of the dataset.
            used for directory structure in S3. Equivalent to a filename
    app_config_name : str
        name of the AppConfig.
    app_config_env : str
        name of the AppConfig environment.
    app_config_app : str
        name of the AppConfig application.
    logger : Logger
        logger object.
            default is the logger object from the module.

    ### Returns
    -------
    filename : str
        full path and name of the file in S3, to be used to store/load dataset data from/to S3.
    exception : Exception
        exception if the dataset is not in the list of allowed datasets.

    ### Example
    -------
    As soon as the AWS Lambda function is prepared and the AppConfig is set it should look something like this:
    ```json
    {
        "data_handler_1": [
            {
                "dataset_1": {
                    "dataset_type": "csv",
                    "api_status": "active"
                },
                "dataset_2": {
                    "dataset_type": "json",
                    "api_status": "active"
                }
            }
        ],
        "data_handler_2": [
            {
                "dataset_1": {
                    "dataset_type": "csv",
                    "api_status": "active"
                },
                "dataset_2": {
                    "dataset_type": "json",
                    "api_status": "inactive"
                }
            }
        ]
    }
    ```

    To validate the dataset existence, you can use the following code:
    ```python
    >>> result = validate_dataset_existance(
        data_handler="data_handler_1",
        dataset="dataset_1",
        app_config_name="MyAppConfigName",
        app_config_env="MyAppConfigEnvironment",
        app_config_app="MyAppConfigApplication"
    )

    >>> print(result)
        data_handler_1/dataset_1.csv
    ```
    """
    # Validate whether the parameters are provided
    required_params = [
        data_handler,
        dataset,
        app_config_name,
        app_config_env,
        app_config_app,
    ]
    for param in required_params:
        if not param:
            error_msg = f"Missing required parameter: {param}"
            logger.error(error_msg)
            raise ValueError(error_msg)

    # Load dataset confiugration from AWS AppConfig
    try:
        dataset_config: bytes = parameters.get_app_config(
            name=app_config_name, environment=app_config_env, application=app_config_app
        )
        dataset_config = json.loads(dataset_config.decode("utf-8"))
    except ClientError as exc:
        error_msg = f"Error loading dataset configuration data. Exception: {exc}"
        logger.error(error_msg)
        raise ValueError(error_msg) from exc

    # Find the correct dataset configuration
    # check if the handler is in the list of allowed handlers
    if data_handler not in dataset_config:
        error_msg = f"validation failed for data_handler {data_handler}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    # check if the dataset is in the list of allowed datasets
    if dataset not in dataset_config[data_handler][0]:
        error_msg = f"validation failed for dataset {dataset}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    # make sure dataset type is defined
    datatype = dataset_config[data_handler][0][dataset]["dataset_type"]
    if not datatype:
        error_msg = f"validation failed for dataset type {datatype}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    # make sure dataset is allowd to be delivered via API
    if dataset_config[data_handler][0][dataset]["api_status"] != "active":
        error_msg = (
            f"validation failed: dataset {dataset} not allowed to be delivered via API"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    # construct filename and return it
    return f"{data_handler}/{dataset}.{datatype}"


def load_file_from_s3(
    data_handler: str,
    dataset: str,
    app_config_name: str,
    app_config_env: str,
    app_config_app: str,
    s3_bucket_output: str,
    region_name: str = "eu-central-1",
    logger: Logger = logger,
):
    """
    Load file from S3 bucket.

    ### Parameters
    ----------
    data_handler : str
        data handler - internal name of the data handler.
            used for directory structure in S3. Equivalent to a directory
    dataset : str
        dataset - internal name of the dataset.
            used for directory structure in S3. Equivalent to a filename
    app_config_name : str
        name of the AppConfig. Needed for loading the dataset configuration.
    app_config_env : str
        name of the AppConfig environment. Needed for loading the dataset configuration.
    app_config_app : str
        name of the AppConfig application. Needed for loading the dataset configuration.
    s3_bucket_output : str
        name of the S3 bucket where the file is stored.
    region_name : str
        name of the region where the S3 bucket is located.
            default is "eu-central-1"
    logger : Logger
        logger object.
            default is the logger object from the module.

    ### Returns
    -------
    fileobj : bytes
        file object as bytes.
    exception : Exception
        exception if the file cannot be loaded from S3.
    """

    # validate dataset
    try:
        filename = validate_dataset_existance(
            data_handler=data_handler,
            dataset=dataset,
            app_config_app=app_config_name,
            app_config_env=app_config_env,
            app_config_name=app_config_app,
            logger=logger,
        )
    except ValueError as exc:
        raise exc

    try:
        boto3.setup_default_session(region_name=region_name)
        s3client = boto3.client("s3", region_name=region_name)
        fileobj = s3client.get_object(Bucket=s3_bucket_output, Key=filename)
    except ClientError as exc:
        error_msg = f"Could not load file {filename} from S3. exception: {exc}"
        logger.exception(error_msg)
        raise ValueError(error_msg) from exc

    return fileobj["Body"].read()


def save_dataset_to_s3(
    data_handler: str,
    dataset: str,
    file_contents: bytes,
    app_config_name: str,
    app_config_env: str,
    app_config_app: str,
    s3_bucket_output: str,
    ssekms_key_id: str,
    region_name: str = "eu-central-1",
    server_side_encryption: str = "aws:kms",
    logger: Logger = logger,
):
    """
    Save file to S3 bucket.

    ### Parameters
    ----------
    data_handler : str
        data handler - internal name of the data handler.
            used for directory structure in S3. Equivalent to a directory
    dataset : str
        dataset - internal name of the dataset.
            used for directory structure in S3. Equivalent to a filename
    file_contents : bytes
        file contents as bytes.
    app_config_name : str
        name of the AppConfig. Needed for loading the dataset configuration.
    app_config_env : str
        name of the AppConfig environment. Needed for loading the dataset configuration.
    app_config_app : str
        name of the AppConfig application. Needed for loading the dataset configuration.
    s3_bucket_output : str
        name of the S3 bucket where the file is stored.
    ssekms_key_id : str
        KMS key ID for server side encryption.
    region_name : str
        name of the region where the S3 bucket is located.
            default is "eu-central-1"
    server_side_encryption : str
        server side encryption type.
            default is "aws:kms"
    logger : Logger
        logger object.
            default is the logger object from the module.

    ### Returns
    -------
    True : bool
        True if the file was saved successfully.
    exception : Exception
        exception if the file cannot be saved to S3.
    """
    # validate dataset
    try:
        filename = validate_dataset_existance(
            data_handler=data_handler,
            dataset=dataset,
            app_config_app=app_config_name,
            app_config_env=app_config_env,
            app_config_name=app_config_app,
            logger=logger,
        )
    except ValueError as exc:
        raise exc

    # write file to OUTPUT folder
    try:
        boto3.setup_default_session(region_name=region_name)
        s3_resouce = boto3.resource("s3")
        s3_resouce.Bucket(s3_bucket_output).put_object(
            Key=filename,
            Body=file_contents,
            ServerSideEncryption=server_side_encryption,
            SSEKMSKeyId=ssekms_key_id,
        )
    except ClientError as exc:
        error_msg = f"Error saving data to S3 as CSV file: {filename}. exception: {exc}"
        logger.exception(error_msg)
        raise ValueError(error_msg) from exc
    return True


def get_s3_presigned_url_readonly(
    data_handler: str,
    dataset: str,
    app_config_name: str,
    app_config_env: str,
    app_config_app: str,
    s3_bucket_output: str,
    region_name: str = "eu-central-1",
    logger: Logger = logger,
):
    """
    Get a presigned URL for the file in S3,
    after validating the dataset against the app config.

    ### Parameters
    ----------
    data_handler : str
        data handler - internal name of the data handler.
            used for directory structure in S3. Equivalent to a directory
    dataset : str
        dataset - internal name of the dataset.
            used for directory structure in S3. Equivalent to a filename
    app_config_name : str
        name of the AppConfig. Needed for loading the dataset configuration.
    app_config_env : str
        name of the AppConfig environment. Needed for loading the dataset configuration.
    app_config_app : str
        name of the AppConfig application. Needed for loading the dataset configuration.
    s3_bucket_output : str
        name of the S3 bucket where the file is stored.
    region_name : str
        name of the region where the S3 bucket is located.
            default is "eu-central-1"
    logger : Logger
        logger object.
            default is the logger object from the module.

    ### Returns
    -------
    presigned_url : str
        presigned URL for the file in S3.
    exception : Exception
        exception if the presigned URL cannot be created.
    """
    # validate dataset
    try:
        filename = validate_dataset_existance(
            data_handler=data_handler,
            dataset=dataset,
            app_config_app=app_config_name,
            app_config_env=app_config_env,
            app_config_name=app_config_app,
            logger=logger,
        )
    except ValueError as exc:
        raise exc

    try:
        # create S3 client
        boto3.setup_default_session(region_name=region_name)
        s3_client = boto3.client("s3", region_name=region_name)
        presigned_url = s3_client.generate_presigned_url(
            ClientMethod="get_object",
            Params={"Bucket": s3_bucket_output, "Key": filename},
            ExpiresIn=120,
        )
    except ClientError as exc:
        error_msg = f"Could not create presigned URL. Exception: {exc}"
        logger.exception(error_msg)
        raise ValueError(error_msg) from exc

    # return the presigned URL
    return presigned_url
