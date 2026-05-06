def template_string(context, template):
    'Return the rendered template content with the current context'
    if not isinstance(context, Context):
        context = Context(context)
    return Template(template).render(context)