def render_template_with_args_in_file(file, template_file_name, **kwargs):
    """
    Get a file and render the content of the template_file_name with kwargs in a file
    :param file: A File Stream to write
    :param template_file_name: path to route with template name
    :param **kwargs: Args to be rendered in template
    """
    template_file_content = "".join(
        codecs.open(
            template_file_name,
            encoding='UTF-8'
        ).readlines()
    )
    template_rendered = string.Template(template_file_content).safe_substitute(**kwargs)
    file.write(template_rendered)