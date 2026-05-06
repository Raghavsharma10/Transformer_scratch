def initialise_loggers(names, log_level=_builtin_logging.WARNING, handler_class=SplitStreamHandler):
    """
    Initialises specified loggers to generate output at the
    specified logging level. If the specified named loggers do not exist,
    they are created.

    :type names: :obj:`list` of :obj:`str`
    :param names: List of logger names.
    :type log_level: :obj:`int`
    :param log_level: Log level for messages, typically
       one of :obj:`logging.DEBUG`, :obj:`logging.INFO`, :obj:`logging.WARN`, :obj:`logging.ERROR`
       or :obj:`logging.CRITICAL`.
       See :ref:`levels`.
    :type handler_class: One of the :obj:`logging.handlers` classes.
    :param handler_class: The handler class for output of log messages,
       for example :obj:`SplitStreamHandler` or :obj:`logging.StreamHandler`.

    Example::

       >>> from array_split import logging
       >>> logging.initialise_loggers(["my_logger",], log_level=logging.INFO)
       >>> logger = logging.getLogger("my_logger")
       >>> logger.info("This is info logging.")
       16:35:09|ARRSPLT| This is info logging.
       >>> logger.debug("Not logged at logging.INFO level.")
       >>>

    """
    frmttr = get_formatter()
    for name in names:
        logr = _builtin_logging.getLogger(name)
        handler = handler_class()
        handler.setFormatter(frmttr)
        logr.addHandler(handler)
        logr.setLevel(log_level)