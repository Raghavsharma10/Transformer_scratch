def _clean(c):
    """
    Nuke docs build target directory so next build is clean.
    """
    if isdir(c.sphinx.target):
        rmtree(c.sphinx.target)