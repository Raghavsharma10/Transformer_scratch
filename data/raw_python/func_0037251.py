def open_stored_file(value, url):
    """
    Deserialize value for a given upload url and return open file.
    Returns None if deserialization fails.
    """
    upload = None
    result = deserialize_upload(value, url)
    filename = result['name']
    storage_class = result['storage']
    if storage_class and filename:
        storage = storage_class()
        if storage.exists(filename):
            upload = storage.open(filename)
            upload.name = os.path.basename(filename)
    return upload