def getRoot():
    """Convenience to return the media root with forward slashes"""
    root = settings.MEDIA_ROOT.replace('\\', '/')
    if not root.endswith('/'):
        root += '/'

    return path.Path(root)