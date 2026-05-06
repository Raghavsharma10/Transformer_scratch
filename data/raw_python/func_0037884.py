def prepare_query_params(**kwargs):
    """
    Prepares given parameters to be used in querystring.
    """
    return [
        (sub_key, sub_value)
        for key, value in kwargs.items()
        for sub_key, sub_value in expand(value, key)
        if sub_value is not None
    ]