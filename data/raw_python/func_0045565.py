def parse_template(template_path, **kwargs):
    """ Load and render template.
        First line of template should contain the subject of email.
        Return tuple with subject and content.
    """
    template = get_template(template_path)
    context = Context(kwargs)
    data = template.render(context).strip()
    subject, content = re.split(r'\r?\n', data, 1)
    return (subject.strip(), content.strip())