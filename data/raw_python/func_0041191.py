def dict2orderedlist(dic, order_list, default='', **kwargs):
    """
    Return a list with dict values ordered by a list of key passed in args.
    """
    result = []
    for key_order in order_list:
        value = get_element(dic, key_order, **kwargs)
        result.append(value if value is not None else default)
    return result