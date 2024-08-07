import os
import logging

def validate_env_settings(
    required_env_vars: list[str]
):
    """
    Validate the required environment variables.
    
    ### Parameters:
    ----------
    required_env_vars : list[str]
        The list of required environment variables.

    ### Returns:
    ----------
    is_valid : bool
        True if all required environment variables are present.
    exception : ValueError
        If any of the required environment variables are missing.
    """
    for env_var in required_env_vars:
        if env_var not in os.environ or os.environ.get(env_var) is None:
            msg = f"Environment variable {env_var} is missing."
            logging.error(msg)
            raise ValueError(msg)
    return True