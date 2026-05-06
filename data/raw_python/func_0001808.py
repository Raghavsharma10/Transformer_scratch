def route_method(method_name, extra_part=False):
    """Custom handler routing decorator.
    Signs a web handler callable with the http method as attribute.

    Args:
        method_name (str): HTTP method name (i.e GET, POST)
        extra_part (bool): Indicates if wrapped callable name should be a part
                           of the actual endpoint.

    Returns:
        A wrapped handler callable.

    examples:
        >>> @route_method('GET')
        ... def method():
        ...     return "Hello!"
        ...
        >>> method.http_method
        'GET'
        >>> method.url_extra_part
        None
    """
    def wrapper(callable_obj):
        if method_name.lower() not in DEFAULT_ROUTES:
            raise HandlerHTTPMethodError(
                'Invalid http method in method: {}'.format(method_name)
            )

        callable_obj.http_method = method_name.upper()

        callable_obj.url_extra_part = callable_obj.__name__ if extra_part\
            else None

        return classmethod(callable_obj)
    return wrapper