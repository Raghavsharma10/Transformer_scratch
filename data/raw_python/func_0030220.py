def get_object_cache_keys(instance):
    """
    Return the cache keys associated with an object.
    """
    if not instance.pk or instance._state.adding:
        return []

    keys = []
    for language in _get_available_languages(instance):
        keys.append(get_urlfield_cache_key(instance.__class__, instance.pk, language))

    return keys