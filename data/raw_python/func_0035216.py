def revoke_access_token(credential):
    """Revoke an access token.
    All future requests with the access token will be invalid.
    Parameters
        credential (OAuth2Credential)
            An authorized user's OAuth 2.0 credentials.
    Raises
        ClientError (APIError)
            Thrown if there was an HTTP error.
    """
    url = build_url(auth.SERVER_HOST, auth.REVOKE_PATH)

    args = {
        'token': credential.access_token,
    }

    response = post(url=url, params=args)

    if response.status_code == codes.ok:
        return

    message = 'Failed to revoke access token: {}.'
    message = message.format(response.reason)
    raise ClientError(response, message)