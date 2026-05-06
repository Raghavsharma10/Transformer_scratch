def abort(http_status_code, exc=None, **kwargs):
    """Raise a HTTPException for the given http_status_code. Attach any keyword
    arguments to the exception for later processing.

    From Flask-Restful. See NOTICE file for license information.
    """
    try:
        sanic.exceptions.abort(http_status_code, exc)
    except sanic.exceptions.SanicException as err:
        err.data = kwargs
        err.exc = exc
        raise err