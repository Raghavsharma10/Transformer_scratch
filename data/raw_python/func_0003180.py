def _get_params_value(params):
    """
        Given an iterator (k1, k2), returns a function that when called
        with an object obj returns a tuple of the form:
        ((k1, obj.parameters[k1]), (k2, obj.parameters[k2]))
    """
    # sort params for consistency
    ord_params = sorted(params)

    def fn(obj):
        l = []
        for p in ord_params:
            try:
                l.append((p, obj.parameters[p]))
            except:
                raise ValueError('{} is not a valid parameter'.format(p))
        return tuple(l)
    return fn