def load_template(filename):
    # type: (str) -> str
    """ Load template from file.

    The templates are part of the package and must be included as
    ``package_data`` in project ``setup.py``.

    Args:
        filename (str):
            The template path. Relative to `peltak` package directory.

    Returns:
        str: The content of the chosen template.
    """
    template_file = os.path.join(PKG_DIR, 'templates', filename)
    with open(template_file) as fp:
        return fp.read()