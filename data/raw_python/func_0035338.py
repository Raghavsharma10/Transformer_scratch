def create_lyft_client(credentials):
    """Create an LyftRidesClient from OAuth 2.0 credentials.
    Parameters
        credentials (dict)
            Dictionary of OAuth 2.0 credentials.
    Returns
        (LyftRidesClient)
            An authorized LyftRidesClient to access API resources.
    """
    oauth2credential = OAuth2Credential(
        client_id=credentials.get('client_id'),
        access_token=credentials.get('access_token'),
        expires_in_seconds=credentials.get('expires_in_seconds'),
        scopes=credentials.get('scopes'),
        grant_type=credentials.get('grant_type'),
        client_secret=credentials.get('client_secret'),
        refresh_token=credentials.get('refresh_token'),
    )
    session = Session(oauth2credential=oauth2credential)
    return LyftRidesClient(session)