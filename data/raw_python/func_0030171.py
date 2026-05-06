def get_library(path=None, root=None, db=None):
    import ambry.library as _l
    """Return the default library for this installation."""

    rc = config(path=path, root=root, db=db )

    return _l.new_library(rc)