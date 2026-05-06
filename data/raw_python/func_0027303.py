def decorator_of_context_manager(ctxt):
    """Converts a context manager into a decorator.

    This decorator will run the decorated function in the context of the
    manager.

    :param ctxt: Context to run the function in.
    :return: Wrapper around the original function.

    """

    def decorator_fn(*outer_args, **outer_kwargs):
        def decorator(fn):
            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                with ctxt(*outer_args, **outer_kwargs):
                    return fn(*args, **kwargs)

            return wrapper

        return decorator

    if getattr(ctxt, "__doc__", None) is None:
        msg = "Decorator that runs the inner function in the context of %s"
        decorator_fn.__doc__ = msg % ctxt
    else:
        decorator_fn.__doc__ = ctxt.__doc__
    return decorator_fn