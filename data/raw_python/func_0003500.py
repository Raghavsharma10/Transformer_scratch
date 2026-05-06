def newthread(template="EPCThread-{0}", **kwds):
    """
    Instantiate :class:`threading.Thread` with an appropriate name.
    """
    if not isinstance(template, str):
        template = '{0}.{1}-{{0}}'.format(template.__module__,
                                          template.__class__.__name__)
    return threading.Thread(
        name=newname(template), **kwds)