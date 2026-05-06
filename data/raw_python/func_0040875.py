def registerLoggers(info, error, debug):
    """
    Add logging functions to this module.

    Functions will be called on various severities (log, error, or debug
    respectively).

    Each function must have the signature:
        fn(message, **kwargs)

    If Python str.format()-style placeholders are in message, kwargs will be
    interpolated.
    """
    global log_info
    global log_error
    global log_debug

    log_info = info
    log_error = error
    log_debug = debug