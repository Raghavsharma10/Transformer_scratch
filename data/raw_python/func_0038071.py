def get_spec(name, http_client=None, config=None):
    """
    :param name: Name of the revision of spec, eg latest or v4
    :param http_client: Requests client used for retrieving specs
    :param config: Spec configuration - see Spec.CONFIG_DEFAULTS
    :return: :class:`bravado_core.spec.Spec`
    """
    http_client = http_client or requests_client.RequestsClient()

    def load_spec():
        loader = Loader(http_client)
        return loader.load_spec(build_spec_url(name))

    spec_dict = cache.get_or_set(build_cache_name(name), load_spec, app_settings.ESI_SPEC_CACHE_DURATION)
    config = dict(CONFIG_DEFAULTS, **(config or {}))
    return Spec.from_dict(spec_dict, build_spec_url(name), http_client, config)