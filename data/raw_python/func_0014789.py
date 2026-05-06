def _urlparse(path):
    """Like urlparse except it assumes 'file://' if no scheme is specified
    """
    url = urlparse.urlparse(path)
    _validate_url(url)
    if not url.scheme or url.scheme == 'file://':
        # Normalize path, and set scheme to "file" if missing
        path = os.path.abspath(
            os.path.expanduser(path))
        url = urlparse.urlparse('file://'+path)
    return url