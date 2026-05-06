def allow_bare_decorator(cls):
    """
    Wrapper for a class decorator which allows for bare decorator and argument syntax
    """

    @wraps(cls)
    def wrapper(*args, **kwargs):
        """"Wrapper for real decorator"""

        # If we weren't only passed a bare class, return class instance
        if kwargs or len(args) != 1 or not isclass(args[0]):  # pylint: disable=no-else-return
            return cls(*args, **kwargs)
        # Otherwise, pass call to instance with default values
        else:
            return cls()(args[0])

    return wrapper