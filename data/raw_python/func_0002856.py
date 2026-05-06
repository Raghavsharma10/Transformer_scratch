def render_template(template_file, template_vars):
    """Render a jinja template."""
    # Load our Jinja templates
    template_dir = "{0}/templates".format(
        os.path.dirname(os.path.abspath(__file__))
    )
    jinja_env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(template_dir),
        trim_blocks=True
    )
    rendered = jinja_env.get_template(template_file).render(template_vars)

    return rendered