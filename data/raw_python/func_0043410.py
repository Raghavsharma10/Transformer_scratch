def cached_instance(model, timeout=None, **filters):
    """ Auto cached model instance.
    """
    if isinstance(model, basestring):
        model = _str_to_model(model)

    cache_key = generate_cache_key(model, **filters)
    return get_cached(cache_key, model.objects.select_related().get, kwargs=filters)