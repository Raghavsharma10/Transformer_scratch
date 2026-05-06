def _format_as_url(path):
    """Make sure ``path`` takes the form of ``/some/url/``."""
    path = sub(r"\.html$", '', path)  # remove any ending .html

    # Make sure it starts/ends with a slash.
    if not path.startswith("/"):
        path = "/{0}".format(path)
    if not path.endswith("/"):
        path = "{0}/".format(path)

    return path