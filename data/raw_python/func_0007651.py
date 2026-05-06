def authenticate(
        client_id=None, client_secret=None,
        client_email=None, private_key=None,
        access_token=None, refresh_token=None,
        account=None, webproperty=None, profile=None,
        identity=None, prefix=None, suffix=None,
        interactive=False, save=False):
    """
    The `authenticate` function will authenticate the user with the Google Analytics API,
    using a variety of strategies: keyword arguments provided to this function, credentials
    stored in in environment variables, credentials stored in the keychain and, finally, by
    asking for missing information interactively in a command-line prompt.

    If necessary (but only if `interactive=True`) this function will also allow the user
    to authorize this Python module to access Google Analytics data on their behalf,
    using an OAuth2 token.
    """

    credentials = oauth.Credentials.find(
        valid=True,
        interactive=interactive,
        prefix=prefix,
        suffix=suffix,
        client_id=client_id,
        client_secret=client_secret,
        client_email=client_email,
        private_key=private_key,
        access_token=access_token,
        refresh_token=refresh_token,
        identity=identity,
        )

    if credentials.incomplete:
        if interactive:
            credentials = authorize(
                client_id=credentials.client_id,
                client_secret=credentials.client_secret,
                save=save,
                identity=credentials.identity,
                prefix=prefix,
                suffix=suffix,
                )
        elif credentials.type == 2:
            credentials = authorize(
                client_email=credentials.client_email,
                private_key=credentials.private_key,
                identity=credentials.identity,
                save=save,
                )
        else:
            raise KeyError("Cannot authenticate: enable interactive authorization, pass a token or use a service account.")
    
    accounts = oauth.authenticate(credentials)
    scope = navigate(accounts, account=account, webproperty=webproperty, profile=profile)
    return scope