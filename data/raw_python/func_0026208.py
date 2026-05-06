def setup_global_logging():
    """
    Initializes capture of stdout/stderr, Python warnings, and exceptions;
    redirecting them to the loggers for the modules from which they originated.
    """

    global global_logging_started

    if not PY3K:
        sys.exc_clear()

    if global_logging_started:
        return

    orig_logger_class = logging.getLoggerClass()
    logging.setLoggerClass(StreamTeeLogger)
    try:
        stdout_logger = logging.getLogger(__name__ + '.stdout')
        stderr_logger = logging.getLogger(__name__ + '.stderr')
    finally:
        logging.setLoggerClass(orig_logger_class)

    stdout_logger.setLevel(logging.INFO)
    stderr_logger.setLevel(logging.ERROR)
    stdout_logger.set_stream(sys.stdout)
    stderr_logger.set_stream(sys.stderr)
    sys.stdout = stdout_logger
    sys.stderr = stderr_logger

    exception_logger = logging.getLogger(__name__ + '.exc')
    sys.excepthook = LoggingExceptionHook(exception_logger)

    logging.captureWarnings(True)

    rawinput = 'input' if PY3K else 'raw_input'
    builtins._original_raw_input = getattr(builtins, rawinput)
    setattr(builtins, rawinput, global_logging_raw_input)

    global_logging_started = True