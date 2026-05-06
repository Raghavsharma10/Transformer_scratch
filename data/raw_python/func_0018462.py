def suppress_logging(log_level=logging.CRITICAL):
    """Suppress logging."""
    logging.disable(log_level)
    yield
    logging.disable(logging.NOTSET)