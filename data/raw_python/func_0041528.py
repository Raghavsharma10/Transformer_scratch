def raise_from(exc, cause):
    """
    Does the same as ``raise LALALA from BLABLABLA`` does in Python 3.
    But works in Python 2 also!

    Please checkout README on https://github.com/9seconds/pep3134
    to get an idea about possible pitfals. But short story is: please
    be pretty carefull with tracebacks. If it is possible, use sys.exc_info
    instead. But in most cases it will work as you expect.
    """

    context_tb = sys.exc_info()[2]

    incorrect_cause = not (
        (isinstance(cause, type) and issubclass(cause, Exception)) or
        isinstance(cause, BaseException) or
        cause is None
    )
    if incorrect_cause:
        raise TypeError("exception causes must derive from BaseException")

    if cause is not None:
        if not getattr(cause, "__pep3134__", False):
            # noinspection PyBroadException
            try:
                raise_(cause)
            except:  # noqa pylint: disable=W0702
                cause = sys.exc_info()[1]
        cause.__fixed_traceback__ = context_tb

    # noinspection PyBroadException
    try:
        raise_(exc)
    except:  # noqa pylint: disable=W0702
        exc = sys.exc_info()[1]

    exc.__original_exception__.__suppress_context__ = True
    exc.__original_exception__.__cause__ = cause
    exc.__original_exception__.__context__ = None

    raise exc