def _execute_if_not_empty(func):
    """ Execute function only if one of input parameters is not empty """
    def wrapper(*args, **kwargs):
        if any(args[1:]) or any(kwargs.items()):
            return func(*args, **kwargs)
    return wrapper