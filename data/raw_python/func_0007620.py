def oauth2_token_exchange(client_id, client_secret, redirect_uri,
                          base_url=OH_BASE_URL, code=None, refresh_token=None):
    """
    Exchange code or refresh token for a new token and refresh token. For the
    first time when a project is created, code is required to generate refresh
    token. Once the refresh token is obtained, it can be used later on for
    obtaining new access token and refresh token. The user must store the
    refresh token to obtain the new access token. For more details visit:
    https://www.openhumans.org/direct-sharing/oauth2-setup/#setup-oauth2-authorization

    :param client_id: This field is the client id of user.
    :param client_secret: This field is the client secret of user.
    :param redirect_uri: This is the user redirect uri.
    :param base_url: It is this URL `https://www.openhumans.org`
    :param code: This field is used to obtain access_token for the first time.
        It's default value is none.
    :param refresh_token: This field is used to obtain a new access_token when
        the token expires.
    """
    if not (code or refresh_token) or (code and refresh_token):
        raise ValueError("Either code or refresh_token must be specified.")
    if code:
        data = {
            'grant_type': 'authorization_code',
            'redirect_uri': redirect_uri,
            'code': code,
        }
    elif refresh_token:
        data = {
            'grant_type': 'refresh_token',
            'refresh_token': refresh_token,
        }
    token_url = urlparse.urljoin(base_url, '/oauth2/token/')
    req = requests.post(
        token_url, data=data,
        auth=requests.auth.HTTPBasicAuth(client_id, client_secret))
    handle_error(req, 200)
    data = req.json()
    return data