def template_links(request, app, template_name, context=None, group=None, force=True):
    '''
    Returns the HTML for the given provider group, using an app and template name.
    This method should not normally be used (use links() instead).  The use of
    this method is when provider need to be called from regular python code instead
    of from within a rendering template environment.
    '''
    if isinstance(app, str):
        app = apps.get_app_config(app)
    if context is None:
        context = {}
    dmp = apps.get_app_config('django_mako_plus')
    template_obj = dmp.engine.get_template_loader(app, create=True).get_mako_template(template_name, force=force)
    return template_obj_links(request, template_obj, context, group)