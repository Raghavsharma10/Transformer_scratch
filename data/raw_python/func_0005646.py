def get_func_params(method, called_params):
    """
    :type method: function
    :type called_params: dict
    :return:
    """
    insp = inspect.getfullargspec(method)
    if not isinstance(called_params, dict):
        raise UserWarning()
    _called_params = called_params.copy()
    params = {}
    arg_count = len(insp.args)
    arg_def_count = len(insp.defaults) if insp.defaults is not None else 0
    for i in range(arg_count):
        arg = insp.args[i]
        if i == 0 and isinstance(method, types.MethodType):
            continue  # skip self argument
        if arg in _called_params:
            params[arg] = _called_params.pop(arg)
        elif i - arg_count + arg_def_count >= 0:
            params[arg] = insp.defaults[i - arg_count + arg_def_count]
        else:
            raise TypeError('Argument "%s" not given' % arg)
    for kwarg in insp.kwonlyargs:
        if kwarg in _called_params:
            params[kwarg] = _called_params.pop(kwarg)
        elif kwarg in insp.kwonlydefaults:
            params[kwarg] = insp.kwonlydefaults[kwarg]
        else:
            raise TypeError('Argument "%s" not given' % kwarg)
    if insp.varkw is None:
        if len(_called_params) > 0:
            raise TypeError('Got unexpected parameter(s): %s'
                            '' % (", ".join(_called_params)))
    else:
        params.update(_called_params)
    return params