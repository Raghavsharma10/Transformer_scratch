def auth_decorator(app=None):
    """
    This decorator wraps a view function so that it is protected when Auth0
    is enabled. This means that any request will be expected to have a signed
    token in the authorization header if the `AUTH0_ENABLED` configuration
    setting is True.

    The authorization header will have the form:

    "authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9....."

    If a request is not properly signed, an attempt is made to provide the
    client with useful error messages. This means that if a request is not
    authorized the underlying view function will not be executed.

    When `AUTH0_ENABLED` is false, this decorator will simply execute the
    decorated view without observing the authorization header.
    :param app:
    :return: Flask view decorator
    """
    def requires_auth(f):
        @functools.wraps(f)
        def decorated(*args, **kwargs):
            # This decorator will only apply with AUTH0_ENABLED set to True.
            if app.config.get('AUTH0_ENABLED', False):
                client_id = app.config.get("AUTH0_CLIENT_ID")
                client_secret = app.config.get("AUTH0_CLIENT_SECRET")
                auth_header = flask.request.headers.get('Authorization', None)
                # Each of these functions will throw a 401 is there is a
                # problem decoding the token with some helpful error message.
                if auth_header:
                    token, profile = decode_header(
                        auth_header, client_id, client_secret)
                else:
                    raise exceptions.NotAuthorizedException()
                # We store the token in the session so that later
                # stages can use it to connect identity and authorization.
                flask.session['auth0_key'] = token
                # Now we need to make sure that on top of having a good token
                # They are authorized, and if not provide an error message
                is_authorized(app.cache, profile['email'])
                is_active(app.cache, token)
            return f(*args, **kwargs)
        return decorated
    return requires_auth