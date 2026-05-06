def render_data(context, templateContent, proxyMode, rendered_data, menukey='menubar'):
    """Render the template"""

    if proxyMode:
        # Update csrf_tokens
        csrf = unicode(context['csrf_token'])
        tag = u'{~__PLUGIT_CSRF_TOKEN__~}'
        rendered_data = unicode(rendered_data, 'utf-8').replace(tag, csrf)

        result = rendered_data  # Render in proxy mode
        menu = None  # Proxy mode plugit do not have menu

    else:
        # Render it
        template = Template(templateContent)
        result = template.render(context)
        menu = _get_node(template, context, menukey)

    return (result, menu)