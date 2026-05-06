def get_ext(url):
    """
    Extract an extension from the url.

    Args:
        url (str): String representation of a url.

    Returns:
        str: Filename extension from a url (without a dot), '' if extension is not present.

    """

    parsed = urllib.parse.urlparse(url)
    root, ext = os.path.splitext(parsed.path)
    return ext.lstrip('.')