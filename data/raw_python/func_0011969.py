def normalize_excludes(excludes):
    """Normalize the excluded directory list."""
    return [os.path.normpath(os.path.abspath(exclude)) for exclude in excludes]