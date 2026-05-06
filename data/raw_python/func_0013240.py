def VerifierMiddleware(verifier):
    """Common wrapper for the authentication modules.
        * Parses the request before passing it on to the authentication module.
        * Sets 'pyoidc' cookie if authentication succeeds.
        * Redirects the user to complete the authentication.
        * Allows the user to retry authentication if it fails.
    :param verifier: authentication module
    """

    @wraps(verifier.verify)
    def wrapper(environ, start_response):
        data = get_post(environ)
        kwargs = dict(urlparse.parse_qsl(data))
        kwargs["state"] = json.loads(urllib.unquote(kwargs["state"]))
        val, completed = verifier.verify(**kwargs)
        if not completed:
            return val(environ, start_response)
        if val:
            set_cookie, cookie_value = verifier.create_cookie(val, "auth")
            cookie_value += "; path=/"

            url = "{base_url}?{query_string}".format(
                base_url="/authorization",
                query_string=kwargs["state"]["query"])
            response = SeeOther(url, headers=[(set_cookie, cookie_value)])
            return response(environ, start_response)
        else:  # Unsuccessful authentication
            url = "{base_url}?{query_string}".format(
                base_url="/authorization",
                query_string=kwargs["state"]["query"])
            response = SeeOther(url)
            return response(environ, start_response)

    return wrapper