def mask_dict_password(dictionary, secret='***'):
    """Replace passwords with a secret in a dictionary."""
    d = dictionary.copy()
    for k in d:
        if 'password' in k:
            d[k] = secret
    return d