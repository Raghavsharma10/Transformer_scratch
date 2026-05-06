def get_es(urls=None, timeout=DEFAULT_TIMEOUT, force_new=False, **settings):
    """Create an elasticsearch `Elasticsearch` object and return it.

    This will aggressively re-use `Elasticsearch` objects with the
    following rules:

    1. if you pass the same argument values to `get_es()`, then it
       will return the same `Elasticsearch` object
    2. if you pass different argument values to `get_es()`, then it
       will return different `Elasticsearch` object
    3. it caches each `Elasticsearch` object that gets created
    4. if you pass in `force_new=True`, then you are guaranteed to get
       a fresh `Elasticsearch` object AND that object will not be
       cached

    :arg urls: list of uris; Elasticsearch hosts to connect to,
        defaults to ``['http://localhost:9200']``
    :arg timeout: int; the timeout in seconds, defaults to 5
    :arg force_new: Forces get_es() to generate a new Elasticsearch
        object rather than pulling it from cache.
    :arg settings: other settings to pass into Elasticsearch
        constructor; See
        `<http://elasticsearch-py.readthedocs.org/>`_ for more details.

    Examples::

        # Returns cached Elasticsearch object
        es = get_es()

        # Returns a new Elasticsearch object
        es = get_es(force_new=True)

        es = get_es(urls=['localhost'])

        es = get_es(urls=['localhost:9200'], timeout=10,
                    max_retries=3)

    """
    # Cheap way of de-None-ifying things
    urls = urls or DEFAULT_URLS

    # v0.7: Check for 'hosts' instead of 'urls'. Take this out in v1.0.
    if 'hosts' in settings:
        raise DeprecationWarning('"hosts" is deprecated in favor of "urls".')

    if not force_new:
        key = _build_key(urls, timeout, **settings)
        if key in _cached_elasticsearch:
            return _cached_elasticsearch[key]

    es = Elasticsearch(urls, timeout=timeout, **settings)

    if not force_new:
        # We don't need to rebuild the key here since we built it in
        # the previous if block, so it's in the namespace. Having said
        # that, this is a little ew.
        _cached_elasticsearch[key] = es

    return es