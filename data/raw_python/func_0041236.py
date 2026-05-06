def _get_arg_names(func):
    ''' this returns the arg names since dictionaries dont guarantee order '''
    args, varargs, keywords, defaults = inspect.getargspec(func)
    return(tuple(args))