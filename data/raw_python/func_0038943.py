def render_template(template, **context):
    """Renders a given template and context.

    :param template: The template name
    :param context: the variables that should be available in the
                    context of the template.
    """
    parts = template.split('/')
    renderer = _get_renderer(parts[:-1])
    return renderer.render(renderer.load_template(parts[-1:][0]), context)