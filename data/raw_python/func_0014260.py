def _check_default(value, parameter, default_chars):
    '''Returns the default if the value is "empty"'''
    # not using a set here because it fails when value is unhashable
    if value in default_chars:
        if parameter.default is inspect.Parameter.empty:
            raise ValueError('Value was empty, but no default value is given in view function for parameter: {} ({})'.format(parameter.position, parameter.name))
        return parameter.default
    return value