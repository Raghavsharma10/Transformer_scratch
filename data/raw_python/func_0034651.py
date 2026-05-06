def verboselogs_module_transform(mod):
    """Make Pylint aware of our custom log levels."""
    if mod.name == 'logging':
        for const in ['NOTICE', 'SPAM', 'SUCCESS', 'VERBOSE']:
            mod.locals[const] = [nodes.Const(const)]