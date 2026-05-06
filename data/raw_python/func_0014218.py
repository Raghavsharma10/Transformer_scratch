def template_obj_links(request, template_obj, context=None, group=None):
    '''
    Returns the HTML for the given provider group, using a template object.
    This method should not normally be used (use links() instead).  The use of
    this method is when provider need to be called from regular python code instead
    of from within a rendering template environment.
    '''
    # the template_obj can be a MakoTemplateAdapter or a Mako Template
    # if our DMP-defined MakoTemplateAdapter, switch to the embedded Mako Template
    template_obj = getattr(template_obj, 'mako_template', template_obj)
    # create a mako context so it seems like we are inside a render
    context_dict = {
        'request': request,
    }
    if isinstance(context, Context):
        for d in context:
            context_dict.update(d)
    elif context is not None:
        context_dict.update(context)
    mako_context = create_mako_context(template_obj, **context_dict)
    return links(mako_context['self'], group=group)