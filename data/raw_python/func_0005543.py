def revrank_dict(dict, key=lambda t: t[1], as_tuple=False):
    """ Reverse sorts a #dict by a given key, optionally returning it as a
        #tuple. By default, the @dict is sorted by it's value.

        @dict: the #dict you wish to sorts
        @key: the #sorted key to use
        @as_tuple: returns result as a #tuple ((k, v),...)

        -> :class:OrderedDict or #tuple
    """
    sorted_list = sorted(dict.items(), key=key, reverse=True)
    return OrderedDict(sorted_list) if not as_tuple else tuple(sorted_list)