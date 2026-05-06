def format_template_file(filename, content):
    """Render a given pystache template file with given content"""

    with open(filename, 'r') as f:
        template = f.read()
        if type(template) != str:
            template = template.decode('utf-8')

    return format_template(template, content)