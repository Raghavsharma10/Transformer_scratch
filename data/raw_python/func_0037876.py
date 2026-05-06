def merge_dicts(dict1, dict2):
    """ Merge two dictionaries and return the result """
    tmp = dict1.copy()
    tmp.update(dict2)
    return tmp