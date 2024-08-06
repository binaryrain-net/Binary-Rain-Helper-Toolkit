import os
import logging

def validate_env_settings(required_env_vars):
    """Validate the required environment variables."""

    for env_var in required_env_vars:
        if env_var not in os.environ or os.environ.get(env_var) is None:
            msg = f"Environment variable {env_var} is missing."
            logging.error(msg)
            raise ValueError(msg)