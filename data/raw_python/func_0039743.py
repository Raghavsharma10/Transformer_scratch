def get_signature(name, func):
    """
    Helper to generate a readable signature for a function
    :param name:
    :param func:
    :return:
    """
    args, varargs, keywords, defaults = inspect.getargspec(func)
    defaults = defaults or []
    posargslen = len(args) - len(defaults)
    if varargs is None and keywords is None:
        sig = name + '('
        sigargs = []
        for idx, arg in enumerate(args):
            if idx < posargslen:
                sigargs.append(arg)
            else:
                default = repr(defaults[idx - posargslen])
                sigargs.append(arg + '=' + default)
        sig += ', '.join(sigargs) + ')'
        return sig
    elif not args and varargs and not keywords and not defaults:
        return name + '(*' + varargs + ')'
    else:
        raise InvalidTask('ape tasks may not use **kwargs')