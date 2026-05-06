def configure_urls(apps, index_view=None, prefixes=None):
    '''
    Configure urls from a list of apps.
    '''
    prefixes = prefixes or {}
    urlpatterns = patterns('')

    if index_view:
        from django.views.generic.base import RedirectView
        urlpatterns += patterns('',
            url(r'^$', RedirectView.as_view(pattern_name=index_view, permanent=False)),
        )

    for app_name in apps:
        app_module = importlib.import_module(app_name)
        if module_has_submodule(app_module, 'urls'):
            module = importlib.import_module("%s.urls" % app_name)
            if not hasattr(module, 'urlpatterns'):
                # Resolver will break if the urls.py file is completely blank.
                continue
            app_prefix = prefixes.get(app_name, app_name.replace("_","-"))
            urlpatterns += patterns(
                '',
                url(
                    r'^%s/' % app_prefix if app_prefix else '',
                    include("%s.urls" % app_name),
                ),
            )


    return urlpatterns