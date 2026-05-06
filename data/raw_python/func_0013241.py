def pyoidcMiddleware(func):
    """Common wrapper for the underlying pyoidc library functions.
    Reads GET params and POST data before passing it on the library and
    converts the response from oic.utils.http_util to wsgi.
    :param func: underlying library function
    """

    def wrapper(environ, start_response):
        data = get_or_post(environ)
        cookies = environ.get("HTTP_COOKIE", "")
        resp = func(request=data, cookie=cookies)
        return resp(environ, start_response)

    return wrapper