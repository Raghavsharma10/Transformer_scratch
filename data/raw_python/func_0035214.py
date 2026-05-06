def request_access_token(
    grant_type,
    client_id=None,
    client_secret=None,
    scopes=None,
    code=None,
    refresh_token=None
):
    """Make an HTTP POST to request an access token.
    Parameters
        grant_type (str)
            Either 'client_credientials' (Client Credentials Grant)
            or 'authorization_code' (Authorization Code Grant).
        client_id (str)
            Your app's Client ID.
        client_secret (str)
            Your app's Client Secret.
        scopes (set)
            Set of permission scopes to request.
            (e.g. {'profile', 'history'})
        code (str)
            The authorization code to switch for an access token.
            Only used in Authorization Code Grant.
        refresh_token (str)
            Refresh token used to get a new access token.
            Only used for Authorization Code Grant.
    Returns
        (requests.Response)
            Successful HTTP response from a 'POST' to request
            an access token.
    Raises
        ClientError (APIError)
            Thrown if there was an HTTP error.
    """
    url = build_url(auth.SERVER_HOST, auth.ACCESS_TOKEN_PATH)

    if isinstance(scopes, set):
        scopes = ' '.join(scopes)

    args = {
        'grant_type': grant_type,
        'client_id': client_id,
        'client_secret': client_secret,
        'scope': scopes,
        'code': code,
        'refresh_token': refresh_token,
    }

    auth_header = HTTPBasicAuth(client_id, client_secret)

    response = post(url=url, auth=auth_header, data=args)

    if response.status_code == codes.ok:
        return response

    message = 'Failed to request access token: {}.'
    message = message.format(response.reason)
    raise ClientError(response, message)