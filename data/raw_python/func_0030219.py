def get_urlfield_cache_key(model, pk, language_code=None):
    """
    The low-level function to get the cache key for a model.
    """
    return 'anyurlfield.{0}.{1}.{2}.{3}'.format(model._meta.app_label, model.__name__, pk, language_code or get_language())