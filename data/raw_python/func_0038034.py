def service_param_string(params):
    """Takes a param section from a metadata class and returns a param string for the service method"""
    p = []
    k = []
    for param in params:
        name = fix_param_name(param['name'])
        if 'required' in param and param['required'] is True:
            p.append(name)
        else:
            if 'default' in param:
                k.append('{name}={default}'.format(name=name, default=param['default']))
            else:
                k.append('{name}=None'.format(name=name))
    p.sort(lambda a, b: len(a) - len(b))
    k.sort()
    a = p + k
    return ', '.join(a)