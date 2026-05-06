def get_database(obj, **params):
    """Get database from given URI/Object."""
    if isinstance(obj, string_types):
        return connect(obj, **params)
    return obj