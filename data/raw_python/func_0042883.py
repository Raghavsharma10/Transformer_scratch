def obj_in_list_always(target_list, obj):
    """
    >>> l = [1,1,1]
    >>> obj_in_list_always(l, 1)
    True
    >>> l.append(2)
    >>> obj_in_list_always(l, 1)
    False
    """
    for item in set(target_list):
        if item is not obj:
            return False
    return True