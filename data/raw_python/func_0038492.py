def _check_methods(cls, subclass):  # pylint: disable=too-many-branches
    """
    Args:
        cls(:py:class:`Plugin`): Parent class
        subclass(:py:class:`Plugin`): Subclass to evaluate

    Returns:
        Result: Named tuple

    Validate abstract methods are defined in subclass
    For error codes see _inspect_class
    """

    for meth, methobj in cls.__abstractmethods__.items():

        # Need to get attribute from dictionary for instance tests to work
        for base in subclass.__mro__:  # pragma: no branch
            if meth in base.__dict__:
                submethobj = base.__dict__[meth]
                break

        # If we found our abstract method, we didn't find anything
        if submethobj is methobj:
            submethobj = UNDEFINED

        # Determine if we have the right method type
        result = None
        bad_arg_spec = 'Argument spec does not match parent for method %s'

        # pylint: disable=deprecated-method
        if isinstance(methobj, property):
            if submethobj is UNDEFINED or not isinstance(submethobj, property):
                result = Result(False, 'Does not contain required property (%s)' % meth, 210)

        elif isinstance(methobj, staticmethod):
            if submethobj is UNDEFINED or not isinstance(submethobj, staticmethod):
                result = Result(False, 'Does not contain required static method (%s)' % meth, 211)
            elif PY26:  # pragma: no cover
                if getfullargspec(methobj.__get__(True)) != \
                   getfullargspec(submethobj.__get__(True)):
                    result = Result(False, bad_arg_spec % meth, 220)
            elif getfullargspec(methobj.__func__) != getfullargspec(submethobj.__func__):
                result = Result(False, bad_arg_spec % meth, 220)

        elif isinstance(methobj, classmethod):
            if submethobj is UNDEFINED or not isinstance(submethobj, classmethod):
                result = Result(False, 'Does not contain required class method (%s)' % meth, 212)
            elif PY26:  # pragma: no cover
                if getfullargspec(methobj.__get__(True).__func__) != \
                   getfullargspec(submethobj.__get__(True).__func__):
                    result = Result(False, bad_arg_spec % meth, 220)
            elif getfullargspec(methobj.__func__) != getfullargspec(submethobj.__func__):
                result = Result(False, bad_arg_spec % meth, 220)

        elif isfunction(methobj):
            if submethobj is UNDEFINED or not isfunction(submethobj):
                result = Result(False, 'Does not contain required method (%s)' % meth, 213)
            elif getfullargspec(methobj) != getfullargspec(submethobj):
                result = Result(False, bad_arg_spec % meth, 220)

        # If it's not a type we're specifically checking, just check for existence
        elif submethobj is UNDEFINED:
            result = Result(False, 'Does not contain required attribute (%s)' % meth, 214)

        if result:
            return result

    return Result(True, None, 0)