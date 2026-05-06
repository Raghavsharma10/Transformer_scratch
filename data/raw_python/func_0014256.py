def convert_boolean(value, parameter, default=False):
    '''
    Converts to boolean (only the first char of the value is used):
        '', '-', None convert to parameter default
        'f', 'F', '0', False always convert to False
        Anything else converts to True.
    '''
    value = _check_default(value, parameter, ( '', '-', None ))
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and len(value) > 0:
        value = value[0]
    return value not in ( 'f', 'F', '0', False, None )