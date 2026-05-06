def accept_arguments(method, number_of_arguments=1):
    """Returns True if the given method will accept the given number of arguments

       method: the method to perform introspection on
       number_of_arguments: the number_of_arguments
    """
    if 'method' in method.__class__.__name__:
        number_of_arguments += 1
        func = getattr(method, 'im_func', getattr(method, '__func__'))
        func_defaults = getattr(func, 'func_defaults', getattr(func, '__defaults__'))
        number_of_defaults = func_defaults and len(func_defaults) or 0
    elif method.__class__.__name__ == 'function':
        func_defaults = getattr(method, 'func_defaults', getattr(method, '__defaults__'))
        number_of_defaults = func_defaults and len(func_defaults) or 0

    coArgCount = getattr(method, 'func_code', getattr(method, '__code__')).co_argcount
    if(coArgCount >= number_of_arguments and coArgCount - number_of_defaults <= number_of_arguments):
        return True

    return False