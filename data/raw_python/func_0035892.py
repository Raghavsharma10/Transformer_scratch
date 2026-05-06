def walk(zk, path='/'):
    """Yields all paths under `path`."""
    children = zk.get_children(path)
    yield path
    for child in children:
        if path == '/':
            subpath = "/%s" % child
        else:
            subpath = "%s/%s" % (path, child)

        for child in walk(zk, subpath):
            yield child