def _build_provider_list():
    """
    Construct the provider registry, using the app settings.
    """
    registry = None
    if appsettings.FLUENT_OEMBED_SOURCE == 'basic':
        registry = bootstrap_basic()
    elif appsettings.FLUENT_OEMBED_SOURCE == 'embedly':
        params = {}
        if appsettings.MICAWBER_EMBEDLY_KEY:
            params['key'] = appsettings.MICAWBER_EMBEDLY_KEY
        registry = bootstrap_embedly(**params)
    elif appsettings.FLUENT_OEMBED_SOURCE == 'noembed':
        registry = bootstrap_noembed(nowrap=1)
    elif appsettings.FLUENT_OEMBED_SOURCE == 'list':
        # Fill list manually in the settings, e.g. to have a fixed set of supported secure providers.
        registry = ProviderRegistry()
        for regex, provider in appsettings.FLUENT_OEMBED_PROVIDER_LIST:
            registry.register(regex, Provider(provider))
    else:
        raise ImproperlyConfigured("Invalid value of FLUENT_OEMBED_SOURCE, only 'basic', 'list', 'noembed' or 'embedly' is supported.")

    # Add any extra providers defined in the settings
    for regex, provider in appsettings.FLUENT_OEMBED_EXTRA_PROVIDERS:
        registry.register(regex, Provider(provider))

    return registry