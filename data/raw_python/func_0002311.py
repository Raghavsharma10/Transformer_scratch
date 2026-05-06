def optimize_logger_level(logger, log_level):
    """
    At runtime, when logging is not active,
    replace the .debug() call with a no-op.
    """
    function_name = _log_functions[log_level]
    if getattr(logger, function_name) is _dummy_log:
        return False

    is_level_logged = logger.isEnabledFor(log_level)
    if not is_level_logged:
        setattr(logger, function_name, _dummy_log)

    return is_level_logged