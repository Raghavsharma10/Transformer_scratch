def encode_scopes(scopes, use_quote=False):
    """
    Creates a string out of a list of scopes.

    :param scopes: A list of scopes
    :param use_quote: Boolean flag indicating whether the string should be quoted
    :return: Scopes as a string
    """
    scopes_as_string = Scope.separator.join(scopes)

    if use_quote:
        return quote(scopes_as_string)
    return scopes_as_string