def authenticate_storage(client_secrets, read_only=False):
    """Authenticates a service account for reading and/or writing on a bucket.
    
    TODO: docstring"""
    if read_only:
        scopes = ['https://www.googleapis.com/auth/devstorage.read_only']
    else:
        scopes = ['https://www.googleapis.com/auth/devstorage.read_write']

    credentials = service_account.Credentials.from_service_account_info(
            client_secrets)
    
    scoped_credentials = credentials.with_scopes(scopes)
    
    return scoped_credentials