def get_template_directories():
    """This function tries to figure out where template directories are located.
    It first inspects the TEMPLATES setting, and if that exists and is not
    empty, uses its values.

    Otherwise, the values from all of the defined DIRS within TEMPLATES are used.

    Returns a set of template directories.

    """
    templates = set()
    for t in settings.TEMPLATES:
        templates = templates.union(set(t.get('DIRS', [])))
    return templates