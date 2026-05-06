def shell(name=None, **attrs):
    """Creates a new :class:`Shell` with a function as callback.  This
    works otherwise the same as :func:`command` just that the `cls`
    parameter is set to :class:`Shell`.
    """
    attrs.setdefault('cls', Shell)
    return click.command(name, **attrs)