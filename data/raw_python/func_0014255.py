def convert_decimal(value, parameter):
    '''
    Converts to decimal.Decimal:
        '', '-', None convert to parameter default
        Anything else uses Decimal constructor
    '''
    value = _check_default(value, parameter, ( '', '-', None ))
    if value is None or isinstance(value, decimal.Decimal):
        return value
    try:
        return decimal.Decimal(value)
    except Exception as e:
        raise ValueError(str(e))