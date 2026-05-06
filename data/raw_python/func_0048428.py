def html_encode(path):
    """Return an HTML encoded Path.

    :param path: ``str``
    :return: ``str``
    """
    if sys.version_info > (3, 2, 0):
        return urllib.parse.quote(utils.ensure_string(path))
    else:
        return urllib.quote(utils.ensure_string(path))