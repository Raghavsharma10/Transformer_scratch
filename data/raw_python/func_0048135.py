def get_error(exc):
    """
    Return the appropriate HTTP status code according to the Exception/Error.
    """

    if isinstance(exc, HTTPError):
        # Returning the HTTP Error code coming from requests module
        return exc.response.status_code, text(exc.response.content)

    if isinstance(exc, Timeout):
        # A timeout is a 408, and it's not a HTTPError (why? dunno).
        return 408, exc

    if isinstance(exc, Http404):
        # 404 is 404
        return 404, exc

    if isinstance(exc, PermissionDenied):
        # Permission denied is 403
        return 403, exc

    if isinstance(exc, SuspiciousOperation):
        # Shouldn't happen, but you never know
        return 400, exc

    # The default error code is 500
    return 500, exc