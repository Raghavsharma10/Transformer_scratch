def arguments_not_none(func):
    """decorator, to check if any arguments are None; raise exception if so"""
    def wrapper(*args, **kwargs):
        for arg in args:
            if arg is None:
                raise NullArgument()
        for arg, val in kwargs.items():
            if val is None:
                raise NullArgument()
        try:
            return func(*args, **kwargs)
        except TypeError as ex:
            if any(statement in ex.args[0] for statement in ['takes exactly',
                                                             'required positional argument']):
                raise NullArgument('Wrong number of arguments provided: ' + str(ex.args[0]))
            else:
                raise
    return wrapper