def _inspect_class(cls, subclass):
    """
    Args:
        cls(:py:class:`Plugin`): Parent class
        subclass(:py:class:`Plugin`): Subclass to evaluate

    Returns:
        Result: Named tuple

    Inspect subclass for inclusion

    Values for errorcode:

        * 0: No error

        Error codes between 0 and 100 are not intended for import

        * 50 Skipload flag is True

        Error codes between 99 and 200 are excluded from import

        * 156: Skipload call returned True

        Error codes 200 and above are malformed classes

        * 210: Missing abstract property
        * 211: Missing abstract static method
        * 212: Missing abstract class method
        * 213: Missing abstract method
        * 214: Missing abstract attribute
        * 220: Argument spec does not match
    """

    if callable(subclass._skipload_):

        result = subclass._skipload_()

        if isinstance(result, tuple):
            skip, msg = result
        else:
            skip, msg = result, None

        if skip:
            return Result(False, msg, 156)

    elif subclass._skipload_:
        return Result(False, 'Skipload flag is True', 50)

    return _check_methods(cls, subclass)