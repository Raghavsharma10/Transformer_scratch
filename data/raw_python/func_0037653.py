def render_template(template_name, context):
    """Render a jinja template"""
    return Environment(
        loader=PackageLoader('remarkable')
    ).get_template(template_name).render(context)