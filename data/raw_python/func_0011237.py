def slice(index, template):
    """Slice a template based on it's positional argument

    Arguments:
        index (int): Position at which to slice
        template (str): Template to slice

    Example:
        >>> slice(0, "{cwd}/{0}/assets/{1}/{2}")
        '{cwd}/{0}'
        >>> slice(1, "{cwd}/{0}/assets/{1}/{2}")
        '{cwd}/{0}/assets/{1}'

    """

    try:
        return re.match("^.*{[%i]}" % index, template).group()
    except AttributeError:
        raise ValueError("Index %i not found in template: %s"
                         % (index, template))