def set_http_caching(request, gateway='crab', region='permanent'):
    """
    Set an HTTP Cache Control header on a request.

    :param pyramid.request.Request request: Request to set headers on.
    :param str gateway: What gateway are we caching for? Defaults to `crab`.
    :param str region: What caching region to use? Defaults to `permanent`.
    :rtype: pyramid.request.Request
    """
    crabpy_exp = request.registry.settings.get('crabpy.%s.cache_config.%s.expiration_time' % (gateway, region), None)
    if crabpy_exp is None:
        return request
    ctime = int(int(crabpy_exp) * 1.05)
    request.response.cache_expires(ctime, public=True)
    return request