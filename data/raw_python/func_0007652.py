def revoke(client_id, client_secret,
        client_email=None, private_key=None,
        access_token=None, refresh_token=None,
        identity=None, prefix=None, suffix=None):

    """
    Given a client id, client secret and either an access token or a refresh token,
    revoke OAuth access to the Google Analytics data and remove any stored credentials
    that use these tokens.
    """

    if client_email and private_key:
        raise ValueError('Two-legged OAuth does not use revokable tokens.')
    
    credentials = oauth.Credentials.find(
        complete=True,
        interactive=False,
        identity=identity,
        client_id=client_id,
        client_secret=client_secret,
        access_token=access_token,
        refresh_token=refresh_token,
        prefix=prefix,
        suffix=suffix,
        )

    retval = credentials.revoke()
    keyring.delete(credentials.identity)
    return retval