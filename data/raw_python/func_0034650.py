def verboselogs_class_transform(cls):
    """Make Pylint aware of our custom logger methods."""
    if cls.name == 'RootLogger':
        for meth in ['notice', 'spam', 'success', 'verbose']:
            cls.locals[meth] = [scoped_nodes.Function(meth, None)]