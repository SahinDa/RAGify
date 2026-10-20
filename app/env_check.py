import os
import logging
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

BASE_REQUIRED_VARS = [
    "CHROMA_API_KEY",
    "CHROMA_TENANT",
    "CHROMA_DATABASE",
    "DATABASE_URL",
]

DEV_REQUIRED_VARS = [
    "NVIDIA_API_KEY",
]

PROD_REQUIRED_VARS = [
    "GEMINI_API_KEY",
]


def validate_env():
    """
    Checks that all required environment variables are set based on ENVIRONMENT.
    Raises RuntimeError listing missing variables.
    """
    env = os.getenv("ENVIRONMENT", "dev").lower()

    required_vars = list(BASE_REQUIRED_VARS)
    if env == "prod":
        required_vars.extend(PROD_REQUIRED_VARS)
    else:
        required_vars.extend(DEV_REQUIRED_VARS)

    missing = [var for var in required_vars if not os.getenv(var)]

    if missing:
        missing_list = ", ".join(missing)
        raise RuntimeError(
            f"Missing required environment variable(s) for '{env}' environment: {missing_list}. "
            f"Please check your .env file or cloud environment configuration."
        )

    logger.info(f"Environment check passed: all required variables for '{env}' are set.")