def path_to_url2(path):
    """
    Convert a path to a file: URL.  The path will be made absolute and have
    quoted path parts.
    """
    path = os.path.normpath(os.path.abspath(path))
    drive, path = os.path.splitdrive(path)
    filepath = path.split(os.path.sep)
    url = '/'.join([urllib.quote(part) for part in filepath])
    if not drive:
        url = url.lstrip('/')
    return 'file:///' + drive + url