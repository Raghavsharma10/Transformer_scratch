def filter_none(list_of_points):
    """
    
    :param list_of_points: 
    :return: list_of_points with None's removed
    """
    remove_elementnone = filter(lambda p: p is not None, list_of_points)
    remove_sublistnone = filter(lambda p: not contains_none(p), remove_elementnone)
    return list(remove_sublistnone)