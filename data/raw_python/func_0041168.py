def normalize_dict(dictionary, **kwargs):
    """
    Given an dict, normalize all of their keys using normalize function.
    """
    result = {}
    if isinstance(dictionary, dict):
        keys = list(dictionary.keys())
        for key in keys:
            result[normalizer(key, **kwargs)] = normalize_dict(dictionary.get(key), **kwargs)
    else:
        result = dictionary
    return result