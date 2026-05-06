def get():
    """Returns the current version without importing pymds."""
    pkgnames = find_packages()

    if len(pkgnames) == 0:
        raise ValueError("Can't find any packages")

    pkgname = pkgnames[0]

    content = open(join(pkgname, '__init__.py')).read()
    c = re.compile(r"__version__ *= *('[^']+'|\"[^\"]+\")")
    m = c.search(content)

    if m is None:
        raise ValueError("Can't find __version__ = ... in __init__.py")

    return m.groups()[0][1: -1]