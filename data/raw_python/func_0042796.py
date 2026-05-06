def normalize_dictionary(data_dict):
    """
    Converts all the keys in "data_dict" to strings. The keys must be
    convertible using str().
    """
    for key, value in data_dict.items():
        if not isinstance(key, str):
            del data_dict[key]
            data_dict[str(key)] = value
    return data_dict