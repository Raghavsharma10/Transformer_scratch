def load_subcommands(group):
    """
    Decorator used to load subcommands from a given ``pkg_resources``
    entrypoint group.  Each function must be appropriately decorated
    with the ``cli_tools`` decorators to be considered an extension.

    :param group: The name of the ``pkg_resources`` entrypoint group.
    """

    def decorator(func):
        adaptor = ScriptAdaptor._get_adaptor(func)
        adaptor._add_extensions(group)
        return func
    return decorator