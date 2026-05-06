def is_empty(value, msg=None, except_=None, inc_zeros=True):
    '''
    is defined, but null or empty like value
    '''
    if hasattr(value, 'empty'):
        # dataframes must check for .empty
        # since they don't define truth value attr
        # take the negative, since below we're
        # checking for cases where value 'is_null'
        value = not bool(value.empty)
    elif inc_zeros and value in ZEROS:
        # also consider 0, 0.0, 0L as 'empty'
        # will check for the negative below
        value = True
    else:
        pass
    _is_null = is_null(value, except_=False)
    result = bool(_is_null or not value)
    if except_:
        return is_true(result, msg=msg, except_=except_)
    else:
        return bool(result)