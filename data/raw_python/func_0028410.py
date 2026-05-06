def get_worksheet_keys(data_dict, result_info_key):
    """Gets sorted keys from the dict, ignoring result_info_key and 'meta' key
    Args:
        data_dict: dict to pull keys from

    Returns:
        list of keys in the dict other than the result_info_key
    """
    keys = set(data_dict.keys())
    keys.remove(result_info_key)
    if 'meta' in keys:
        keys.remove('meta')
    return sorted(keys)