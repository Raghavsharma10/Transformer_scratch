def pattern_from_template(templates, name):
    """Return pattern for name

    Arguments:
        templates (dict): Current templates
        name (str): Name of name

    """

    if name not in templates:
        echo("No template named \"%s\"" % name)
        sys.exit(1)

    return templates[name]