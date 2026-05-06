def convert_datetime(value, parameter):
    '''
    Converts to datetime.datetime:
        '', '-', None convert to parameter default
        The first matching format in settings.DATETIME_INPUT_FORMATS converts to datetime
    '''
    value = _check_default(value, parameter, ( '', '-', None ))
    if value is None or isinstance(value, datetime.datetime):
        return value
    for fmt in settings.DATETIME_INPUT_FORMATS:
        try:
            return datetime.datetime.strptime(value, fmt)
        except (ValueError, TypeError):
            continue
    raise ValueError("`{}` does not match a format in settings.DATETIME_INPUT_FORMATS".format(value))