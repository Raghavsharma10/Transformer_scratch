def dmp_paths_for_app(app_name, pattern_kwargs=None, pretty_app_name=None):
    '''Utility function that creates the default patterns for an app'''
    dmp = apps.get_app_config('django_mako_plus')
    # Because these patterns are subpatterns within the app's resolver,
    # we don't include the /app/ in the pattern -- it's already been
    # handled by the app's resolver.
    #
    # Also note how the each pattern below defines the four kwargs--
    # either as 1) a regex named group or 2) in kwargs.
    return [
        # page.function/urlparams
        dmp_path(
            r'^(?P<dmp_page>[_a-zA-Z0-9\-]+)\.(?P<dmp_function>[_a-zA-Z0-9\.\-]+)/(?P<dmp_urlparams>.+?)/?$',
            merge_dicts({
                'dmp_app': app_name or dmp.options['DEFAULT_APP'],
            }, pattern_kwargs),
            'DMP /{}/page.function/urlparams'.format(pretty_app_name),
            app_name,
        ),

        # page.function
        dmp_path(
            r'^(?P<dmp_page>[_a-zA-Z0-9\-]+)\.(?P<dmp_function>[_a-zA-Z0-9\.\-]+)/?$',
            merge_dicts({
                'dmp_app': app_name or dmp.options['DEFAULT_APP'],
                'dmp_urlparams': '',
            }, pattern_kwargs),
            'DMP /{}/page.function'.format(pretty_app_name),
            app_name,
        ),

        # page/urlparams
        dmp_path(
            r'^(?P<dmp_page>[_a-zA-Z0-9\-]+)/(?P<dmp_urlparams>.+?)/?$',
            merge_dicts({
                'dmp_app': app_name or dmp.options['DEFAULT_APP'],
                'dmp_function': 'process_request',
            }, pattern_kwargs),
            'DMP /{}/page/urlparams'.format(pretty_app_name),
            app_name,
        ),

        # page
        dmp_path(
            r'^(?P<dmp_page>[_a-zA-Z0-9\-]+)/?$',
            merge_dicts({
                'dmp_app': app_name or dmp.options['DEFAULT_APP'],
                'dmp_function': 'process_request',
                'dmp_urlparams': '',
            }, pattern_kwargs),
            'DMP /{}/page'.format(pretty_app_name),
            app_name,
        ),

        # empty
        dmp_path(
            r'^$',
            merge_dicts({
                'dmp_app': app_name or dmp.options['DEFAULT_APP'],
                'dmp_function': 'process_request',
                'dmp_urlparams': '',
                'dmp_page': dmp.options['DEFAULT_PAGE'],
            }, pattern_kwargs),
            'DMP /{}'.format(pretty_app_name),
            app_name,
        ),
    ]