import logging
import sys

def setup_logging():
    """
    Configures logging once at app setup. Call this before anything else runs.
    """

    logging.basicConfig(
        level = logging.INFO,
        format = "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers = [
            logging.StreamHandler(sys.stdout)
        ]
    )