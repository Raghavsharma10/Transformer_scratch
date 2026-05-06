def convert_int(value, parameter):
    '''
    Converts to int or float:
        '', '-', None convert to parameter default
        Anything else uses int() or float() constructor
    '''
    value = _check_default(value, parameter, ( '', '-', None ))
    if value is None or isinstance(value, int):
        return value
    try:
        return int(value)
    except Exception as e:
        raise ValueError(str(e))