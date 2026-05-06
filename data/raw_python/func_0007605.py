def oauth2_auth_url_cli(redirect_uri=None, client_id=None,
                        base_url=OH_BASE_URL):
    """
    Command line function for obtaining the Oauth2 url.
    For more information visit
    :func:`oauth2_auth_url<ohapi.api.oauth2_auth_url>`.
    """
    result = oauth2_auth_url(redirect_uri, client_id, base_url)
    print('The requested URL is : \r')
    print(result)