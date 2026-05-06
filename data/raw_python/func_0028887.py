def keys_to_snake_case(camel_case_dict):
    """
    Make a copy of a dictionary with all keys converted to snake case. This is just calls to_snake_case on
    each of the keys in the dictionary and returns a new dictionary.

    :param camel_case_dict: Dictionary with the keys to convert.
    :type camel_case_dict: Dictionary.

    :return: Dictionary with the keys converted to snake case.
    """
    return dict((to_snake_case(key), value) for (key, value) in camel_case_dict.items())