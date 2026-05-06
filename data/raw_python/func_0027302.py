def retry(exception_cls, max_tries=10, sleep=0.05):
    """Decorator for retrying a function if it throws an exception.

    :param exception_cls: an exception type or a parenthesized tuple of exception types
    :param max_tries: maximum number of times this function can be executed. Must be at least 1.
    :param sleep: number of seconds to sleep between function retries

    """

    assert max_tries > 0

    def with_max_retries_call(delegate):
        for i in xrange(0, max_tries):
            try:
                return delegate()
            except exception_cls:
                if i + 1 == max_tries:
                    raise
                time.sleep(sleep)

    def outer(fn):
        is_generator = inspect.isgeneratorfunction(fn)

        @functools.wraps(fn)
        def retry_fun(*args, **kwargs):
            return with_max_retries_call(lambda: fn(*args, **kwargs))

        @functools.wraps(fn)
        def retry_generator_fun(*args, **kwargs):
            def get_first_item():
                results = fn(*args, **kwargs)
                for first_result in results:
                    return [first_result], results
                return [], results

            cache, generator = with_max_retries_call(get_first_item)

            for item in cache:
                yield item

            for item in generator:
                yield item

        if not is_generator:
            # so that qcore.inspection.get_original_fn can retrieve the original function
            retry_fun.fn = fn
            # Necessary for pickling of Cythonized functions to work. Cython's __reduce__
            # method always returns the original name of the function.
            retry_fun.__reduce__ = lambda: fn.__name__
            return retry_fun
        else:
            retry_generator_fun.fn = fn
            retry_generator_fun.__reduce__ = lambda: fn.__name__
            return retry_generator_fun

    return outer