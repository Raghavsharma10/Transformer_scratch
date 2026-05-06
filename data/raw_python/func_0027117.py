def catchable_exceptions(exceptions):
    """Returns True if exceptions can be caught in the except clause.

    The exception can be caught if it is an Exception type or a tuple of
    exception types.

    """
    if isinstance(exceptions, type) and issubclass(exceptions, BaseException):
        return True

    if (
        isinstance(exceptions, tuple)
        and exceptions
        and all(issubclass(it, BaseException) for it in exceptions)
    ):
        return True

    return False