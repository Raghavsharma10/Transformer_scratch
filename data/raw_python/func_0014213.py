def render_template_for_path(request, path, context=None, use_cache=True, def_name=None):
    '''
    Convenience method that directly renders a template, given a direct path to it.
    '''
    return get_template_for_path(path, use_cache).render(context, request, def_name)