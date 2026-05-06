def is_null(value, msg=None, except_=None):
    '''
    ie, "is not defined"
    '''
    # dataframes, even if empty, are not considered null
    value = False if hasattr(value, 'empty') else value
    result = bool(
        value is None or
        value != value or
        repr(value) == 'NaT')
    if except_:
        return is_true(result, msg=msg, except_=except_)
    else:
        return bool(result)