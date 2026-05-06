def rbigint_to_string(obj):
    """ Recursively converts big integers (|>2**53-1|) to strings

        @obj: Any python object

        -> @obj, with any big integers converted to #str objects
    """
    if isinstance(obj, (str, bytes)) or not obj:
        # the input is the desired one, return as is
        return obj
    elif hasattr(obj, 'items'):
        # the input is a dict {}
        for k, item in obj.items():
            obj[k] = rbigint_to_string(item)
        return obj
    elif hasattr(obj, '__iter__'):
        # the input is iterable
        is_tuple = isinstance(obj, tuple)
        if is_tuple:
            obj = list(obj)
        for i, item in enumerate(obj):
            obj[i] = rbigint_to_string(item)
        return obj if not is_tuple else tuple(obj)
    return bigint_to_string(obj)