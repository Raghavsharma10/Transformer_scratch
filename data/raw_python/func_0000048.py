def unstub(*objs):
    """Unstubs all stubbed methods and functions

    If you don't pass in any argument, *all* registered mocks and
    patched modules, classes etc. will be unstubbed.

    Note that additionally, the underlying registry will be cleaned.
    After an `unstub` you can't :func:`verify` anymore because all
    interactions will be forgotten.
    """

    if objs:
        for obj in objs:
            mock_registry.unstub(obj)
    else:
        mock_registry.unstub_all()