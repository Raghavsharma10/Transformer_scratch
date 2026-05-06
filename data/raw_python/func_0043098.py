def cors_wrapper(func):
    """
    Decorator for CORS
    :param func:  Flask method that handles requests and returns a response
    :return: Same, but with permissive CORS headers set
    """
    def _setdefault(obj, key, value):
        if value == None:
            return
        obj.setdefault(key, value)

    def output(*args, **kwargs):
        response = func(*args, **kwargs)
        headers = response.headers
        _setdefault(headers, "Access-Control-Allow-Origin", "*")
        _setdefault(headers, "Access-Control-Allow-Headers", flask.request.headers.get("Access-Control-Request-Headers"))
        _setdefault(headers, "Access-Control-Allow-Methods", flask.request.headers.get("Access-Control-Request-Methods"))
        _setdefault(headers, "Content-Type", "application/json")
        _setdefault(headers, "Strict-Transport-Security", "max-age=31536000; includeSubDomains; preload")
        return response

    output.provide_automatic_options = False
    output.__name__ = func.__name__
    return output