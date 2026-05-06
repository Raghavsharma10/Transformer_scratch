def config(path=None, root=None, db=None):
    """Return the default run_config object for this installation."""
    import ambry.run
    return ambry.run.load(path=path, root=root, db=db)