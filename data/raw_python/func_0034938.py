def preparse(template_text, lookup=None):
    """ Do any special processing of a template, including recognizing the templating language
        and resolving file: references, then return an appropriate wrapper object.

        Currently Tempita and Python string interpolation are supported.
        `lookup` is an optional callable that resolves any ambiguous template path.
    """
    # First, try to resolve file: references to their contents
    template_path = None
    try:
        is_file = template_text.startswith("file:")
    except (AttributeError, TypeError):
        pass # not a string
    else:
        if is_file:
            template_path = template_text[5:]
            if template_path.startswith('/'):
                template_path = '/' + template_path.lstrip('/')
            elif template_path.startswith('~'):
                template_path = os.path.expanduser(template_path)
            elif lookup:
                template_path = lookup(template_path)

            with closing(open(template_path, "r")) as handle:
                template_text = handle.read().rstrip()

    if hasattr(template_text, "__engine__"):
        # Already preparsed
        template = template_text
    else:
        if template_text.startswith("{{"):
            import tempita  # only on demand

            template = tempita.Template(template_text, name=template_path)
            template.__engine__ = "tempita"
        else:
            template = InterpolationTemplate(template_text)

        template.__file__ = template_path

    template.__text__ = template_text
    return template