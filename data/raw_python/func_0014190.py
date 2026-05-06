def create_view_for_template(app_name, template_name):
    '''
    Creates a view function for templates (used whe a view.py file doesn't exist but the .html does)
    Raises TemplateDoesNotExist if the template doesn't exist.
    '''
    # ensure the template exists
    apps.get_app_config('django_mako_plus').engine.get_template_loader(app_name).get_template(template_name)
    # create the view function
    def template_view(request, *args, **kwargs):
        # not caching the template object (getting it each time) because Mako has its own cache
        dmp = apps.get_app_config('django_mako_plus')
        template = dmp.engine.get_template_loader(app_name).get_template(template_name)
        return template.render_to_response(request=request, context=kwargs)
    template_view.view_type = 'template'
    return template_view