def authenticate_compute(client_secrets):
    """Authenticates a service account for the compute engine.
    
    TODO: docstring"""
    scopes = ['https://www.googleapis.com/auth/compute']

    credentials = ServiceAccountCredentials.from_json_keyfile_dict(
            client_secrets, scopes=scopes)
    
    return credentials