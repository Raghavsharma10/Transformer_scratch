def setup_logging(progname, level, handlers, **kwargs):
    """Setup logging to stdout (stream), syslog, or stackdriver.

    Attaches handler(s) and sets log level to the root logger.

    Example usage:

        import logging
        from ulogger import setup_logging

        setup_logging('my_awesome_program', 'INFO', ['stream'])

        logging.info('ohai')

    Args:
        progname (str): Name of program.
        level (str): Threshold for when to log.
        handlers (list): Desired handlers, default 'stream',
            supported: 'syslog', 'stackdriver', 'stream'.
        **kwargs (optional): Keyword arguments to pass to handlers. See
            handler documentation for more information on available
            kwargs.
    """
    for h in handlers:
        if h == 'stream':
            handler = _setup_default_handler(progname, **kwargs)
        else:
            handler_module_path = 'ulogger.{}'.format(h)
            try:
                handler_module = import_module(
                    handler_module_path, package='ulogger')
            except ImportError:
                msg = 'Unsupported log handler: "{}".'.format(h)
                raise exceptions.ULoggerError(msg)

            try:
                get_handler = getattr(handler_module, 'get_handler')
            except AttributeError:
                msg = '"get_handler" function not implemented for "{}".'
                raise exceptions.ULoggerError(msg.format(h))

            handler = get_handler(progname, **kwargs)
        logging.getLogger('').addHandler(handler)

    level = logging.getLevelName(level)
    logging.getLogger('').setLevel(level)