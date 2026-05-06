def get_client(key, project):
    """Gets a `Client` object (required by the other functions).
    
    TODO: docstring"""
    cred = get_storage_credentials(key)
    return storage.Client(project=project, credentials=cred)