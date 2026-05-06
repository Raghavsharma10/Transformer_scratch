def drop_reserved_params(params):
    """ Drops reserved params """
    from nefertari import RESERVED_PARAMS
    params = params.copy()
    for reserved_param in RESERVED_PARAMS:
        if reserved_param in params:
            params.pop(reserved_param)
    return params