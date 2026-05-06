def initialize_providers(cls):
        '''Initializes the providers (called from dmp app ready())'''
        dmp = apps.get_app_config('django_mako_plus')
        # regular content providers
        cls.CONTENT_PROVIDERS = []
        for provider_settings in dmp.options[cls.SETTINGS_KEY]:
            # import the class for this provider
            assert 'provider' in provider_settings, "Invalid entry in settings.py: CONTENT_PROVIDERS item must have 'provider' key"
            provider_cls = import_string(provider_settings['provider'])
            # combine options from all of its bases, then from settings.py
            options = {}
            for base in reversed(inspect.getmro(provider_cls)):
                options.update(getattr(base, 'DEFAULT_OPTIONS', {}))
            options.update(provider_settings)
            # add to the list
            if options['enabled']:
                pe = ProviderEntry(provider_cls, options)
                pe.options['template_cache_key'] = '_dmp_provider_{}_'.format(id(pe))
                cls.CONTENT_PROVIDERS.append(pe)