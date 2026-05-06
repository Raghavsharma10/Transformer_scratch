def django_include(context, template_name, **kwargs):
    '''
    Mako tag to include a Django template withing the current DMP (Mako) template.
    Since this is a Django template, it is search for using the Django search
    algorithm (instead of the DMP app-based concept).
    See https://docs.djangoproject.com/en/2.1/topics/templates/.

    The current context is sent to the included template, which makes all context
    variables available to the Django template. Any additional kwargs are added
    to the context.
    '''
    try:
        djengine = engines['django']
    except KeyError as e:
        raise TemplateDoesNotExist("Django template engine not configured in settings, so template cannot be found: {}".format(template_name)) from e
    djtemplate = djengine.get_template(template_name)
    djcontext = {}
    djcontext.update(context)
    djcontext.update(kwargs)
    return djtemplate.render(djcontext, context['request'])