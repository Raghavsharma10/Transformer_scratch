def _iget(key, lookup_dict):
    """
    Case-insensitive search for `key` within keys of `lookup_dict`.
    """
    for k, v in lookup_dict.items():
        if k.lower() == key.lower():
            return v
    return None