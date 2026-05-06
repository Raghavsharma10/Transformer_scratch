def get_template_loader(app, subdir='templates'):
    '''
    Convenience method that calls get_template_loader() on the DMP
    template engine instance.
    '''
    dmp = apps.get_app_config('django_mako_plus')
    return dmp.engine.get_template_loader(app, subdir, create=True)