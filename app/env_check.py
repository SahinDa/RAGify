import os
import logging

logger = logging.getLogger(__name__)

REQUIRED_ENV_VARS = [
    "NVIDIA_API_KEY",
    "CHROMA_API_KEY",
    "CHROMA_TENANT",
    "CHROMA_DATABASE",
    "DATABASE_URL",
]


def validate_env():
    """
    Checks that all required environment variables are set.
    Raises a clear RuntimeError listing exactly what's missing,
    rather than letting the app crash later with a confusing
    library-internal error.
    """
    missing = [var for var in REQUIRED_ENV_VARS if not os.getenv(var)]

    if missing:
        missing_list = ", ".join(missing)
        raise RuntimeError(
            f"Missing required environment variable(s): {missing_list}. "
            f"Check your .env file."
        )

    logger.info("All required environment variables are set.")