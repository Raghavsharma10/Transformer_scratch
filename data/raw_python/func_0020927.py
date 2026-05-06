def query_string_parser(search_pattern):
    """Elasticsearch query string parser."""
    if not hasattr(current_oaiserver, 'query_parser'):
        query_parser = current_app.config['OAISERVER_QUERY_PARSER']
        if isinstance(query_parser, six.string_types):
            query_parser = import_string(query_parser)
        current_oaiserver.query_parser = query_parser
    return current_oaiserver.query_parser('query_string', query=search_pattern)