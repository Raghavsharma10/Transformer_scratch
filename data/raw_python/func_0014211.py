def get_template_loader_for_path(path, use_cache=True):
    '''
    Convenience method that calls get_template_loader_for_path() on the DMP
    template engine instance.
    '''
    dmp = apps.get_app_config('django_mako_plus')
    return dmp.engine.get_template_loader_for_path(path, use_cache)