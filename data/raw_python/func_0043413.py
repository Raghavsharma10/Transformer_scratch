def generate_cache_key(cached, **kwargs):
    """ Auto generate cache key for model or queryset
    """

    if isinstance(cached, QuerySet):
        key = str(cached.query)

    elif isinstance(cached, (Model, ModelBase)):
        key = '%s.%s:%s' % (cached._meta.app_label,
                            cached._meta.module_name,
                            ','.join('%s=%s' % item for item in kwargs.iteritems()))

    else:
        raise AttributeError("Objects must be queryset or model.")

    if not key:
        raise Exception('Cache key cannot be empty.')

    key = clean_cache_key(key)
    return key