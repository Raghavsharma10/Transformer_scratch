def import_oauth2_credentials(filename=STORAGE_FILENAME):
    """Import OAuth 2.0 session credentials from storage file.
    Parameters
        filename (str)
            Name of storage file.
    Returns
        credentials (dict)
            All your app credentials and information
            imported from the configuration file.
    """
    with open(filename, 'r') as storage_file:
        storage = safe_load(storage_file)

    # depending on OAuth 2.0 grant_type, these values may not exist
    client_secret = storage.get('client_secret')
    refresh_token = storage.get('refresh_token')

    credentials = {
        'access_token': storage['access_token'],
        'client_id': storage['client_id'],
        'client_secret': client_secret,
        'expires_in_seconds': storage['expires_in_seconds'],
        'grant_type': storage['grant_type'],
        'refresh_token': refresh_token,
        'scopes': storage['scopes'],
    }

    return credentials