def app_resolver(app_name=None, pattern_kwargs=None, name=None):
    '''
    Registers the given app_name with DMP and adds convention-based
    url patterns for it.

    This function is meant to be called in a project's urls.py.
    '''
    urlconf = URLConf(app_name, pattern_kwargs)
    resolver = re_path(
        '^{}/?'.format(app_name) if app_name is not None else '',
        include(urlconf),
        name=urlconf.app_name,
    )
    # this next line is a workaround for Django's URLResolver class not having
    # a `name` attribute, which is expected in Django's technical_404.html.
    resolver.name = getattr(resolver, 'name', name or app_name)
    return resolver