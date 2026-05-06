def jwt_optional(fn):
    """
    If you decorate a view with this, it will check the request for a valid
    JWT and put it into the Flask application context before calling the view.
    If no authorization header is present, the view will be called without the
    application context being changed. Other authentication errors are not
    affected. For example, if an expired JWT is passed in, it will still not
    be able to access an endpoint protected by this decorator.

    :param fn: The view function to decorate
    """
    @wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            jwt_data = _decode_jwt_from_headers()
            ctx_stack.top.jwt = jwt_data
        except (NoAuthorizationError, InvalidHeaderError):
            pass
        return fn(*args, **kwargs)
    return wrapper