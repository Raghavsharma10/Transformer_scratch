def wrap(v):
    """
    WRAP AS Data OBJECT FOR DATA PROCESSING: https://github.com/klahnakoski/mo-dots/tree/dev/docs
    :param v:  THE VALUE TO WRAP
    :return:  Data INSTANCE
    """

    type_ = _get(v, CLASS)

    if type_ is dict:
        m = object.__new__(Data)
        _set(m, SLOT, v)
        return m
    elif type_ is none_type:
        return Null
    elif type_ is list:
        return FlatList(v)
    elif type_ in generator_types:
        return FlatList(list(unwrap(vv) for vv in v))
    else:
        return v