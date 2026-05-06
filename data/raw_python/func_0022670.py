def _find_module(name, path=None):
    """
    Alternative to `imp.find_module` that can also search in subpackages.
    """

    parts = name.split('.')

    for part in parts:
        if path is not None:
            path = [path]

        fh, path, descr = imp.find_module(part, path)
        if fh is not None and part != parts[-1]:
            fh.close()

    return fh, path, descr