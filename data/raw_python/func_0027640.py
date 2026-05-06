def dict_merge(*dict_list):
    """recursively merges dict's. not just simple a['key'] = b['key'], if
    both a and bhave a key who's value is a dict then dict_merge is called
    on both values and the result stored in the returned dictionary.
    """
    result = collections.defaultdict(dict)
    dicts_items = itertools.chain(*[six.iteritems(d or {}) for d in dict_list])

    for key, value in dicts_items:
        src = result[key]
        if isinstance(src, dict) and isinstance(value, dict):
            result[key] = dict_merge(src, value)
        elif isinstance(src, dict) or isinstance(src, six.text_type):
            result[key] = value
        elif hasattr(src, "__iter__") and hasattr(value, "__iter__"):
            result[key] += value
        else:
            result[key] = value

    return dict(result)