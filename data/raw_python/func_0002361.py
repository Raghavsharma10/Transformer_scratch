def jwt_required(fn):
    """
    If you decorate a view with this, it will ensure that the requester has a
    valid JWT before calling the actual view.

    :param fn: The view function to decorate
    """
    @wraps(fn)
    def wrapper(*args, **kwargs):
        jwt_data = _decode_jwt_from_headers()
        ctx_stack.top.jwt = jwt_data
        return fn(*args, **kwargs)
    return wrapper