def enable_thread_logging(exception_callback=None):
    """
    Monkey-patch the threading.Thread class with our own LoggedThread. Any subsequent imports of threading.Thread
    will reference LoggedThread instead.
    """
    global logged_thread_enabled, Thread
    LoggedThread.exception_callback = exception_callback
    Thread = threading.Thread = LoggedThread
    logged_thread_enabled = True