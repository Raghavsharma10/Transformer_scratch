def cache_spec(name, spec):
    """
    Cache the spec dict
    :param name: Version name
    :param spec: Spec dict
    :return: True if cached
    """
    return cache.set(build_cache_name(name), spec, app_settings.ESI_SPEC_CACHE_DURATION)