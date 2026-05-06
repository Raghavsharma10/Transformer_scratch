def get_uri_obj(uri, storage_args={}):
    """
    Retrieve the underlying storage object based on the URI (i.e., scheme).

    :param str uri: URI to get storage object for
    :param dict storage_args: Keyword arguments to pass to the underlying storage object
    """

    if isinstance(uri, BaseURI): return uri
    uri_obj = None

    o = urlparse(uri)
    for storage in STORAGES:
        uri_obj = storage.parse_uri(o, storage_args=storage_args)
        if uri_obj is not None:
            break
    #end for
    if uri_obj is None:
        raise TypeError('<{}> is an unsupported URI.'.format(uri))

    return uri_obj