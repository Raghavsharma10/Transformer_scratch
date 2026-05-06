def get_runconfig(path=None, root=None, db=None):
    """Load the main configuration files and accounts file.

    Debprecated. Use load()
    """

    return load(path, root=root, db=db)