def get_storage_credentials(key, read_only=False):
    """Authenticates a service account for reading and/or writing on a bucket.
    
    This uses the `google.oauth2.service_account` module to obtain "scoped
    credentials". These can be used with the `google.storage` module.

    TODO: docstring"""
    if read_only:
        scopes = ['https://www.googleapis.com/auth/devstorage.read_only']
    else:
        scopes = ['https://www.googleapis.com/auth/devstorage.read_write']

    credentials = service_account.Credentials.from_service_account_info(key)
    
    scoped_credentials = credentials.with_scopes(scopes)
    
    return scoped_credentials