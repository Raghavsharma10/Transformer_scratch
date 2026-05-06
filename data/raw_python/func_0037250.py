def deserialize_upload(value, url):
    """
    Restore file and name and storage from serialized value and the upload url.
    """
    result = {'name': None, 'storage': None}
    try:
        result = signing.loads(value, salt=url)
    except signing.BadSignature:
        # TODO: Log invalid signature
        pass
    else:
        try:
            result['storage'] = get_storage_class(result['storage'])
        except (ImproperlyConfigured, ImportError):
            # TODO: Log invalid class
            result = {'name': None, 'storage': None}
    return result