def log_exception(level='warning', tb_skip=2):
    """
    Send an exception and traceback to the logger.
    
    This function is used in cases where an exception is handled safely but
    nevertheless should generate a descriptive error message. An extra line
    is inserted into the stack trace indicating where the exception was caught.
    
    Parameters
    ----------
    level : str
        See ``set_log_level`` for options.
    tb_skip : int
        The number of traceback entries to ignore, prior to the point where
        the exception was caught. The default is 2.
    """
    stack = "".join(traceback.format_stack()[:-tb_skip])
    tb = traceback.format_exception(*sys.exc_info())
    msg = tb[0]  # "Traceback (most recent call last):"
    msg += stack
    msg += "  << caught exception here: >>\n"
    msg += "".join(tb[1:]).rstrip()
    logger.log(logging_types[level], msg)