def log_exception(log="reusables", message=None, exceptions=(Exception, ),
                  level=logging.ERROR, show_traceback=True):
    """
    Wrapper. Log the traceback to any exceptions raised. Possible to raise
    custom exception.

    .. code :: python

        @reusables.log_exception()
        def test():
            raise Exception("Bad")

        # 2016-12-26 12:38:01,381 - reusables   ERROR  Exception in test - Bad
        # Traceback (most recent call last):
        #     File "<input>", line 1, in <module>
        #     File "reusables\wrappers.py", line 200, in wrapper
        #     raise err
        # Exception: Bad

    Message format options: {func} {err} {args} {kwargs}

    :param exceptions: types of exceptions to catch
    :param log: log name to use
    :param message: message to use in log
    :param level: logging level
    :param show_traceback: include full traceback or just error message
    """
    def func_wrapper(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            msg = message if message else "Exception in '{func}': {err}"
            if not message:
                msg = _add_args(msg, *args, **kwargs)

            try:
                return func(*args, **kwargs)
            except exceptions as err:
                my_logger = (logging.getLogger(log) if isinstance(log, str)
                             else log)
                my_logger.log(level, msg.format(func=func.__name__,
                                                err=str(err),
                                                args=args, kwargs=kwargs),
                              exc_info=show_traceback)
                raise err
        return wrapper
    return func_wrapper