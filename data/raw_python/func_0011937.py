def _is_exception(exceptions, before_token, after_token, token):
    """Predicate for whether the open token is in an exception context

    :arg exceptions: list of strings or None
    :arg before_token: the text of the function up to the token delimiter
    :arg after_token: the text of the function after the token delimiter
    :arg token: the token (only if we're looking at a close delimiter

    :returns: bool

    """
    if not exceptions:
        return False
    for s in exceptions:
        if before_token.endswith(s):
            return True
        if s in token:
            return True
    return False