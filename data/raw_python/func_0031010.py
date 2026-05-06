def get_logger(name='', log_stream=None, log_file=None,
               quiet=False, verbose=False):
    """Convenience function for getting a logger."""

    # configure root logger
    log_level = logging.INFO
    if quiet:
        log_level = logging.WARNING
    elif verbose:
        log_level = logging.DEBUG

    if log_stream is None:
        log_stream = sys.stdout

    new_logger = configure_logger(name, log_stream=log_stream,
                                  log_file=log_file, log_level=log_level)

    return new_logger