def http_basic_auth(request):
    """
    Extracts the credentials of a client using HTTP Basic Auth.

    Expects the ``client_id`` to be the username and the ``client_secret`` to
    be the password part of the Authorization header.

    :param request: The incoming request
    :type request: oauth2.web.Request

    :return: A tuple in the format of (<CLIENT ID>, <CLIENT SECRET>)`
    :rtype: tuple
    """
    auth_header = request.header("authorization")

    if auth_header is None:
        raise OAuthInvalidError(error="invalid_request",
                                explanation="Authorization header is missing")

    auth_parts = auth_header.strip().encode("latin1").split(None)

    if auth_parts[0].strip().lower() != b'basic':
        raise OAuthInvalidError(
            error="invalid_request",
            explanation="Provider supports basic authentication only")

    client_id, client_secret = b64decode(auth_parts[1]).split(b':', 1)

    return client_id.decode("latin1"), client_secret.decode("latin1")