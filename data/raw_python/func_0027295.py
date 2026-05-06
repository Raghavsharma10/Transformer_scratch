def Remote(path=None, loader=Notebook, **globals):
    """A remote notebook finder.  Place a `*` into a url
    to generalize the finder.  It returns a context manager
    """

    class Remote(RemoteMixin, loader):
        ...

    return Remote(path=path, **globals)