def set_secret_key(token):
    """
    Initializes a Authentication and sets it as the new default global authentication.
    It also performs some checks before saving the authentication.

    :Example

    >>> # Expected format for secret key:
    >>> import payplug
    >>> payplug.set_secret_key('sk_test_somerandomcharacters')

    :param token: your secret token (live or sandbox)
    :type token: string
    """
    if not isinstance(token, string_types):
        raise exceptions.ConfigurationError('Expected string value for token.')

    config.secret_key = token