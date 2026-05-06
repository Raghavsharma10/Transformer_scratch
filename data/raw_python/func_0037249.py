def serialize_upload(name, storage, url):
    """
    Serialize uploaded file by name and storage. Namespaced by the upload url.
    """
    if isinstance(storage, LazyObject):
        # Unwrap lazy storage class
        storage._setup()
        cls = storage._wrapped.__class__
    else:
        cls = storage.__class__
    return signing.dumps({
        'name': name,
        'storage': '%s.%s' % (cls.__module__, cls.__name__)
    }, salt=url)