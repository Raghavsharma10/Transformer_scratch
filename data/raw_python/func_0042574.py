def copy_tpl(template_file, dst, template_vars):
    """This supports jinja2 templates. Please feel encouraged to use the
    template framework of your choosing.
    jinja2 docu: http://jinja.pocoo.org/docs/2.9/

    :param template_file:
    :param dst:
    :param template_vars: dictionary containing key, values used in the template
    """
    create_dir(dst)

    # load template
    template_loader = jinja2.FileSystemLoader(searchpath='/')
    template_env = jinja2.Environment(loader=template_loader,
                                      keep_trailing_newline=True)
    template = template_env.get_template(template_file)

    # render and write to file
    template.stream(template_vars).dump(dst)