def log_raise(log, err_str, err_type=RuntimeError):
    """Log an error message and raise an error.

    Arguments
    ---------
    log : `logging.Logger` object
    err_str : str
        Error message to be logged and raised.
    err_type : `Exception` object
        Type of error to raise.

    """
    log.error(err_str)
    # Make sure output is flushed
    # (happens automatically to `StreamHandlers`, but not `FileHandlers`)
    for handle in log.handlers:
        handle.flush()
    # Raise given error
    raise err_type(err_str)