def oauth_token_exchange_cli(client_id, client_secret, redirect_uri,
                             base_url=OH_BASE_URL, code=None,
                             refresh_token=None):
    """
    Command line function for obtaining the refresh token/code.
    For more information visit
    :func:`oauth2_token_exchange<ohapi.api.oauth2_token_exchange>`.
    """
    print(oauth2_token_exchange(client_id, client_secret, redirect_uri,
                                base_url, code, refresh_token))