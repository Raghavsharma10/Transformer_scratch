def is_excluded(root, excludes):
    """Check if the directory is in the exclude list.

    Note: by having trailing slashes, we avoid common prefix issues, like
          e.g. an exlude "foo" also accidentally excluding "foobar".
    """
    root = os.path.normpath(root)
    for exclude in excludes:
        if root == exclude:
            return True
    return False