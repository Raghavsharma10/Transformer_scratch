def item_path_or(default, keys, dict_or_obj):
    """
    Optional version of item_path with a default value. keys can be dict keys or object attributes, or a combination
    :param default:
    :param keys: List of keys or dot-separated string
    :param dict_or_obj: A dict or obj
    :return:
    """
    if not keys:
        raise ValueError("Expected at least one key, got {0}".format(keys))
    resolved_keys = keys.split('.') if isinstance(str, keys) else keys
    current_value = dict_or_obj
    for key in resolved_keys:
        current_value = prop_or(default, key, default_to({}, current_value))
    return current_value