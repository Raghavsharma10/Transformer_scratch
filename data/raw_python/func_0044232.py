def requests_get(url):
    """ Run :func:`requests.get` in a ``cached()`` wrapper.

    The cache wrapper uses the default timeout (environment variable
    ``PYTHON_FTR_CACHE_TIMEOUT``, 3 days by default).

    It is used in :func:`ftr_process`.
    """

    LOGGER.info(u'Fetching %s…', url)
    return requests.get(url)