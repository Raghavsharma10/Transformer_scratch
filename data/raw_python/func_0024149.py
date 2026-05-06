def merge_truthy(*dicts):
    """Merge multiple dictionaries, keeping the truthy values in case of key collisions.

    Accepts any number of dictionaries, or any other object that returns a 2-tuple of
    key and value pairs when its `.items()` method is called.

    If a key exists in multiple dictionaries passed to this function, the values from the latter
    dictionary is kept. If the value of the latter dictionary does not evaluate to True, then
    the value of the previous dictionary is kept.

    >>> merge_truthy({'a': 1, 'c': 4}, {'a': None, 'b': 2}, {'b': 3})
    {'a': 1, 'b': 3, 'c': 4}
    """
    merged = {}
    for d in dicts:
        for k, v in d.items():
            merged[k] = v or merged.get(k, v)
    return merged