def get_template_for_path(path, use_cache=True):
    '''
    Convenience method that retrieves a template given a direct path to it.
    '''
    dmp = apps.get_app_config('django_mako_plus')
    app_path, template_name = os.path.split(path)
    return dmp.engine.get_template_loader_for_path(app_path, use_cache=use_cache).get_template(template_name)