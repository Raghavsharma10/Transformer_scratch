def _make_context(context=None):
    """Create the namespace of items already pre-imported when using shell.

    Accepts a dict with the desired namespace as the key, and the object as the
    value.
    """
    namespace = {'db': db, 'session': db.session}
    namespace.update(_iter_context())

    if context is not None:
        namespace.update(context)

    return namespace