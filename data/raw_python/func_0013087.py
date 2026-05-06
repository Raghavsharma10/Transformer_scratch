def _gql(cls, query_string, *args, **kwds):
    """Run a GQL query."""
    from .query import gql  # Import late to avoid circular imports.
    return gql('SELECT * FROM %s %s' % (cls._class_name(), query_string),
               *args, **kwds)