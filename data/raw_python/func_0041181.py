def _get_element_by_names(source, names):
    """
    Given a dict and path '/' or '.' separated. Digs into de dict to retrieve
    the specified element.

    Args:
        source (dict): set of nested objects in which the data will be searched
        path (list): list of attribute names
    """

    if source is None:
        return source

    else:
        if names:
            head, *rest = names
            if isinstance(source, dict) and head in source:
                return _get_element_by_names(source[head], rest)
            elif isinstance(source, list) and head.isdigit():
                return _get_element_by_names(source[int(head)], rest)
            elif not names[0]:
                pass
            else:
                source = None
        return source