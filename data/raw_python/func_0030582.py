def get_handler(progname, address=None, proto=None, facility=None,
                fmt=None, datefmt=None, **_):
    """Helper function to create a Syslog handler.

    See `ulogger.syslog.SyslogHandlerBuilder` for arguments and
    supported keyword arguments.

    Returns:
        (obj): Instance of `logging.SysLogHandler`
    """
    builder = SyslogHandlerBuilder(
        progname, address=address, proto=proto, facility=facility,
        fmt=fmt, datefmt=datefmt)
    return builder.get_handler()