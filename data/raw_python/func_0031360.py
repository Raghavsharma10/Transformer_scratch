def get_compute_credentials(key):
    """Authenticates a service account for the compute engine.

    This uses the `oauth2client.service_account` module. Since the `google`
    Python package does not support the compute engine (yet?), we need to make
    direct HTTP requests. For that we need authentication tokens. Obtaining
    these based on the credentials provided by the `google.auth2` module is
    much more cumbersome than using the `oauth2client` module.

    See:
    - https://cloud.google.com/iap/docs/authentication-howto
    - https://developers.google.com/identity/protocols/OAuth2ServiceAccount
    
    TODO: docstring"""
    scopes = ['https://www.googleapis.com/auth/compute']

    credentials = ServiceAccountCredentials.from_json_keyfile_dict(
        key, scopes=scopes)
    
    return credentials