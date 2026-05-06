def dict_diff(d1, d2, no_key='<KEYNOTFOUND>'):
    # type: (DictUpperBound, DictUpperBound, str) -> Dict
    """Compares two dictionaries

    Args:
        d1 (DictUpperBound): First dictionary to compare
        d2 (DictUpperBound): Second dictionary to compare
        no_key (str): What value to use if key is not found Defaults to '<KEYNOTFOUND>'.

    Returns:
        Dict: Comparison dictionary

    """
    d1keys = set(d1.keys())
    d2keys = set(d2.keys())
    both = d1keys & d2keys
    diff = {k: (d1[k], d2[k]) for k in both if d1[k] != d2[k]}
    diff.update({k: (d1[k], no_key) for k in d1keys - both})
    diff.update({k: (no_key, d2[k]) for k in d2keys - both})
    return diff