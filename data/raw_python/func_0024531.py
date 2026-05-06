def patch(make_pool=_default_make_pool):
    """Monkey-patches httplib2.Http to be httplib2shim.Http.

    This effectively makes all clients of httplib2 use urlilb3. It's preferable
    to specify httplib2shim.Http explicitly where you can, but this can be
    useful in situations where you do not control the construction of the http
    object.

    Args:
        make_pool: A function that returns a urllib3.Pool-like object. This
            allows you to specify special arguments to your connection pool if
            needed. By default, this will create a urllib3.PoolManager with
            SSL verification enabled using the certifi certificates.
    """
    setattr(httplib2, '_HttpOriginal', httplib2.Http)
    httplib2.Http = Http
    Http._make_pool = make_pool