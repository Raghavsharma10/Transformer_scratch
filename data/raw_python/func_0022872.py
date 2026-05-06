def _check_conversion(key, valid_dict):
    """Check for existence of key in dict, return value or raise error"""
    if key not in valid_dict and key not in valid_dict.values():
        # Only show users the nice string values
        keys = [v for v in valid_dict.keys() if isinstance(v, string_types)]
        raise ValueError('value must be one of %s, not %s' % (keys, key))
    return valid_dict[key] if key in valid_dict else key