def log_view(func):
    """
    Helpful while debugging Selenium unittests.
    e.g.: server response an error in AJAX requests
    """

    @functools.wraps(func)
    def view_logger(*args, **kwargs):
        log.debug("call view %r", func.__name__)
        try:
            response = func(*args, **kwargs)
        except Exception as err:
            log.error("view exception: %s", err)
            traceback.print_exc(file=sys.stderr)
            raise

        log.debug("Response: %s", response)
        return response

    return view_logger