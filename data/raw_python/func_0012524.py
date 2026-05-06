def time_it(log=None, message=None, append=None):
    """
    Wrapper. Time the amount of time it takes the execution of the function
    and print it.

    If log is true, make sure to set the logging level of 'reusables' to INFO
    level or lower.

    .. code:: python

        import time
        import reusables

        reusables.add_stream_handler('reusables')

        @reusables.time_it(log=True, message="{seconds:.2f} seconds")
        def test_time(length):
            time.sleep(length)
            return "slept {0}".format(length)

        result = test_time(5)
        # 2016-11-09 16:59:39,935 - reusables.wrappers  INFO      5.01 seconds

        print(result)
        # slept 5

    Message format options: {func} {seconds} {args} {kwargs}

    :param log: log as INFO level instead of printing
    :param message: string to format with total time as the only input
    :param append: list to append item too
    """
    def func_wrapper(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Can't use nonlocal in 2.x
            msg = (message if message else
                   "Function '{func}' took a total of {seconds} seconds")
            if not message:
                msg = _add_args(msg, *args, **kwargs)

            time_func = (time.perf_counter if python_version >= (3, 3)
                         else time.time)
            start_time = time_func()
            try:
                return func(*args, **kwargs)
            finally:
                total_time = time_func() - start_time

                time_string = msg.format(func=func.__name__,
                                         seconds=total_time,
                                         args=args, kwargs=kwargs)
                if log:
                    my_logger = logging.getLogger(log) if isinstance(log, str)\
                                else logger
                    my_logger.info(time_string)
                else:
                    print(time_string)
                if isinstance(append, list):
                    append.append(total_time)
        return wrapper
    return func_wrapper