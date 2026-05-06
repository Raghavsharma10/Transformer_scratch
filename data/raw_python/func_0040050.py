def is_same_dict(d1, d2):
    """Test two dictionary is equal on values. (ignore order)
    """
    for k, v in d1.items():
        if isinstance(v, dict):
            is_same_dict(v, d2[k])
        else:
            assert d1[k] == d2[k]

    for k, v in d2.items():
        if isinstance(v, dict):
            is_same_dict(v, d1[k])
        else:
            assert d1[k] == d2[k]