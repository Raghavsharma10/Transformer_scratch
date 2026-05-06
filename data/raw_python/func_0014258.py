def convert_date(value, parameter):
    '''
    Converts to datetime.date:
        '', '-', None convert to parameter default
        The first matching format in settings.DATE_INPUT_FORMATS converts to datetime
    '''
    value = _check_default(value, parameter, ( '', '-', None ))
    if value is None or isinstance(value, datetime.date):
        return value
    for fmt in settings.DATE_INPUT_FORMATS:
        try:
            return datetime.datetime.strptime(value, fmt).date()
        except (ValueError, TypeError):
            continue
    raise ValueError("`{}` does not match a format in settings.DATE_INPUT_FORMATS".format(value))