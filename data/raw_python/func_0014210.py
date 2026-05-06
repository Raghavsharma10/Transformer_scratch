def render_template(request, app, template_name, context=None, subdir="templates", def_name=None):
    '''
    Convenience method that directly renders a template, given the app and template names.
    '''
    return get_template(app, template_name, subdir).render(context, request, def_name)