def timer(logger=None, level=logging.INFO,
          fmt="function %(function_name)s execution time: %(execution_time).3f",
          *func_or_func_args, **timer_kwargs):
    """ Function decorator displaying the function execution time

    All kwargs are the arguments taken by the Timer class constructor.

    """
    # store Timer kwargs in local variable so the namespace isn't polluted
    # by different level args and kwargs

    def wrapped_f(f):
        @functools.wraps(f)
        def wrapped(*args, **kwargs):
            with Timer(**timer_kwargs) as t:
                out = f(*args, **kwargs)
            context = {
                'function_name': f.__name__,
                'execution_time': t.elapsed,
            }
            if logger:
                logger.log(
                    level,
                    fmt % context,
                    extra=context)
            else:
                print(fmt % context)
            return out
        return wrapped
    if (len(func_or_func_args) == 1
            and isinstance(func_or_func_args[0], collections.Callable)):
        return wrapped_f(func_or_func_args[0])
    else:
        return wrapped_f