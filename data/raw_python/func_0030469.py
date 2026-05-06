def uri_to_iri_parts(path, query, fragment):
    r"""
    Converts a URI parts to corresponding IRI parts in a given charset.

    Examples for URI versus IRI:

    :param path: The path of URI to convert.
    :param query: The query string of URI to convert.
    :param fragment: The fragment of URI to convert.
    """
    path = url_unquote(path, '%/;?')
    query = url_unquote(query, '%;/?:@&=+,$#')
    fragment = url_unquote(fragment, '%;/?:@&=+,$#')
    return path, query, fragment