"""Centralized logger setup."""
import logging
import os

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

_loggers: dict[str, logging.Logger] = {}


def setup_logger(name: str) -> logging.Logger:
    if name in _loggers:
        return _loggers[name]
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "%(asctime)s  %(levelname)-7s  %(name)s — %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
        logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))
    _loggers[name] = logger
    return logger
