def glitter_head(context):
    """
    Template tag which renders the glitter CSS and JavaScript. Any resources
    which need to be loaded should be added here. This is only shown to users
    with permission to edit the page.
    """
    user = context.get('user')
    rendered = ''
    template_path = 'glitter/include/head.html'

    if user is not None and user.is_staff:
        template = context.template.engine.get_template(template_path)
        rendered = template.render(context)

    return rendered