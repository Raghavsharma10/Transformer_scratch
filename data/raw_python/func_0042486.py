def parser():
    """Return search query parser."""
    query_parser = current_app.config['COLLECTIONS_QUERY_PARSER']
    if isinstance(query_parser, six.string_types):
        query_parser = import_string(query_parser)
        return query_parser