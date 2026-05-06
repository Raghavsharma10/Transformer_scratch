def refresh_access_token(credential):
    """Use a refresh token to request a new access token.
    Not suported for access tokens obtained via Implicit Grant.
    Parameters
        credential (OAuth2Credential)
            An authorized user's OAuth 2.0 credentials.
    Returns
        (Session)
            A new Session object with refreshed OAuth 2.0 credentials.
    Raises
        LyftIllegalState (APIError)
            Raised if OAuth 2.0 grant type does not support
            refresh tokens.
    """
    if credential.grant_type == auth.AUTHORIZATION_CODE_GRANT:
        response = request_access_token(
            grant_type=auth.REFRESH_TOKEN,
            client_id=credential.client_id,
            client_secret=credential.client_secret,
            refresh_token=credential.refresh_token,
        )

        oauth2credential = OAuth2Credential.make_from_response(
            response=response,
            grant_type=credential.grant_type,
            client_id=credential.client_id,
            client_secret=credential.client_secret,
        )

        return Session(oauth2credential=oauth2credential)

    elif credential.grant_type == auth.CLIENT_CREDENTIAL_GRANT:
        response = request_access_token(
            grant_type=auth.CLIENT_CREDENTIAL_GRANT,
            client_id=credential.client_id,
            client_secret=credential.client_secret,
            scopes=credential.scopes,
        )

        oauth2credential = OAuth2Credential.make_from_response(
            response=response,
            grant_type=credential.grant_type,
            client_id=credential.client_id,
            client_secret=credential.client_secret,
        )

        return Session(oauth2credential=oauth2credential)

    message = '{} Grant Type does not support Refresh Tokens.'
    message = message.format(credential.grant_type)
    raise LyftIllegalState(message)