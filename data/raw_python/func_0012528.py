def retry_it(exceptions=(Exception, ), tries=10, wait=0, handler=None,
             raised_exception=ReusablesError, raised_message=None):
    """
    Retry a function if an exception is raised, or if output_check returns
    False.

    Message format options: {func} {args} {kwargs}

    :param exceptions: tuple of exceptions to catch
    :param tries: number of tries to retry the function
    :param wait: time to wait between executions in seconds
    :param handler: function to check if output is valid, must return bool 
    :param raised_exception: default is ReusablesError
    :param raised_message: message to pass to raised exception
    """
    def func_wrapper(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            msg = (raised_message if raised_message
                   else "Max retries exceeded for function '{func}'")
            if not raised_message:
                msg = _add_args(msg, *args, **kwargs)
            try:
                result = func(*args, **kwargs)
            except exceptions:
                if tries:
                    if wait:
                        time.sleep(wait)
                    return retry_it(exceptions=exceptions, tries=tries-1,
                                    handler=handler,
                                    wait=wait)(func)(*args, **kwargs)
                if raised_exception:
                    exc = raised_exception(msg.format(func=func.__name__,
                                           args=args, kwargs=kwargs))
                    exc.__cause__ = None
                    raise exc
            else:
                if handler:
                    if not handler(result):
                        return retry_it(exceptions=exceptions, tries=tries - 1,
                                        handler=handler,
                                        wait=wait)(func)(*args, **kwargs)
                return result
        return wrapper
    return func_wrapper