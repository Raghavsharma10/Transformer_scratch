def es_required(fun):
    """Wrap a callable and return None if ES_DISABLED is False.

    This also adds an additional `es` argument to the callable
    giving you an ElasticSearch instance to use.

    """
    @wraps(fun)
    def wrapper(*args, **kw):
        if getattr(settings, 'ES_DISABLED', False):
            log.debug('Search disabled for %s.' % fun)
            return

        return fun(*args, es=get_es(), **kw)
    return wrapper