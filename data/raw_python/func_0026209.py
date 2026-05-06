def teardown_global_logging():
    """Disable global logging of stdio, warnings, and exceptions."""

    global global_logging_started
    if not global_logging_started:
        return

    stdout_logger = logging.getLogger(__name__ + '.stdout')
    stderr_logger = logging.getLogger(__name__ + '.stderr')
    if sys.stdout is stdout_logger:
        sys.stdout = sys.stdout.stream
    if sys.stderr is stderr_logger:
        sys.stderr = sys.stderr.stream

    # If we still have an unhandled exception go ahead and handle it with the
    # replacement excepthook before deleting it
    exc_type, exc_value, exc_traceback = sys.exc_info()
    if exc_type is not None:
        sys.excepthook(exc_type, exc_value, exc_traceback)
    del exc_type
    del exc_value
    del exc_traceback
    if not PY3K:
        sys.exc_clear()

    del sys.excepthook
    logging.captureWarnings(False)

    rawinput = 'input' if PY3K else 'raw_input'
    if hasattr(builtins, '_original_raw_input'):
        setattr(builtins, rawinput, builtins._original_raw_input)
        del builtins._original_raw_input

    global_logging_started = False