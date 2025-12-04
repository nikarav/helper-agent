import logging


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the given name.

    :param name: The name of the logger.
    :return: The logger object.
    """

    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(logging.INFO)
        console_handler = logging.StreamHandler()

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        logger.propagate = False
    return logger


def set_log_level(logger: logging.Logger, level: int | str) -> None:
    """
    Set the log level for a logger.

    :param logger: The logger object.
    :param level: The log level to set.
    :return: None
    """
    logger.setLevel(level)
