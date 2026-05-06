def trace(enter=False, exit=True):
    """
    This decorator prints entry and exit message when
    the decorated method is called, as well as call
    arguments, result and thrown exception (if any).

    :param enter: indicates whether entry message should be printed.
    :param exit: indicates whether exit message should be printed.
    :return: decorated function.

    """

    def decorate(fn):
        @inspection.wraps(fn)
        def new_fn(*args, **kwargs):
            name = fn.__module__ + "." + fn.__name__
            if enter:
                print(
                    "%s(args = %s, kwargs = %s) <-" % (name, repr(args), repr(kwargs))
                )
            try:
                result = fn(*args, **kwargs)
                if exit:
                    print(
                        "%s(args = %s, kwargs = %s) -> %s"
                        % (name, repr(args), repr(kwargs), repr(result))
                    )
                return result
            except Exception as e:
                if exit:
                    print(
                        "%s(args = %s, kwargs = %s) -> thrown %s"
                        % (name, repr(args), repr(kwargs), str(e))
                    )
                raise

        return new_fn

    return decorate