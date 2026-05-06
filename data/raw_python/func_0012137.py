def local_settings(settings, prefix):
    """Localizes the settings for the dotted prefix.
    For example, if the prefix where 'xyz'::

        {'xyz.foo': 'bar', 'other': 'something'}

    Would become::

        {'foo': 'bar'}

    Note, that non-prefixed items are left out and the prefix is dropped.
    """
    prefix = "{}.".format(prefix)
    new_settings = {k[len(prefix):]: v for k, v in settings.items()
                    if k.startswith(prefix)}
    return new_settings