def request_body(request):
    """
    Extracts the credentials of a client from the
    *application/x-www-form-urlencoded* body of a request.

    Expects the client_id to be the value of the ``client_id`` parameter and
    the client_secret to be the value of the ``client_secret`` parameter.

    :param request: The incoming request
    :type request: oauth2.web.Request

    :return: A tuple in the format of `(<CLIENT ID>, <CLIENT SECRET>)`
    :rtype: tuple
    """
    client_id = request.post_param("client_id")
    if client_id is None:
        raise OAuthInvalidError(error="invalid_request",
                                explanation="Missing client identifier")

    client_secret = request.post_param("client_secret")
    if client_secret is None:
        raise OAuthInvalidError(error="invalid_request",
                                explanation="Missing client credentials")

    return client_id, client_secret