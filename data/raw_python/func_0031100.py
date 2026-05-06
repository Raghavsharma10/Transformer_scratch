def configure_settings(settings, environment_settings=True):
    '''
    Given a settings object, run automatic configuration of all
    the apps in INSTALLED_APPS.
    '''
    changes = 1
    iterations = 0

    while changes:
        changes = 0
        app_names = ['django_autoconfig'] + list(settings['INSTALLED_APPS'])
        if environment_settings:
            app_names.append('django_autoconfig.environment_settings')
        for app_name in app_names:
            import django_autoconfig.contrib
            if autoconfig_module_exists(app_name):
                module = importlib.import_module("%s.autoconfig" % (app_name,))
            elif app_name in django_autoconfig.contrib.CONTRIB_CONFIGS:
                module = django_autoconfig.contrib.CONTRIB_CONFIGS[app_name]
            else:
                continue
            changes += merge_dictionaries(
                settings,
                getattr(module, 'SETTINGS', {}),
                template_special_case=True,
            )
            changes += merge_dictionaries(
                settings,
                getattr(module, 'DEFAULT_SETTINGS', {}),
                only_defaults=True,
            )
            for relationship in getattr(module, 'RELATIONSHIPS', []):
                changes += relationship.apply_changes(settings)

        if iterations >= MAX_ITERATIONS:
            raise ImproperlyConfigured(
                'Autoconfiguration could not reach a consistent state'
            )
        iterations += 1
    LOGGER.debug("Autoconfiguration took %d iterations.", iterations)