def make_android_api_method(req_method, secure=True, version=0):
    """Turn an AndroidApi's method into a function that builds the request,
    sends it, then passes the response to the actual method. Should be used
    as a decorator.
    """
    def outer_func(func):
        def inner_func(self, **kwargs):
            req_url = self._build_request_url(secure, func.__name__, version)
            req_func = self._build_request(req_method, req_url, params=kwargs)
            response = req_func()
            func(self, response)
            return response
        return inner_func
    return outer_func