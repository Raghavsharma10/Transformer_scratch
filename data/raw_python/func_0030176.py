def get_handler(progname, fmt=None, datefmt=None, project_id=None,
                credentials=None, debug_thread_worker=False, **_):
    """Helper function to create a Stackdriver handler.

    See `ulogger.stackdriver.CloudLoggingHandlerBuilder` for arguments
    and supported keyword arguments.

    Returns:
        (obj): Instance of `google.cloud.logging.handlers.
                            CloudLoggingHandler`
    """
    builder = CloudLoggingHandlerBuilder(
        progname, fmt=fmt, datefmt=datefmt, project_id=project_id,
        credentials=credentials, debug_thread_worker=debug_thread_worker)
    return builder.get_handler()