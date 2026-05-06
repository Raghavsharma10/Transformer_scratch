def get_element(source, path, separator=r'[/.]'):
    """
    Given a dict and path '/' or '.' separated. Digs into de dict to retrieve
    the specified element.

    Args:
        source (dict): set of nested objects in which the data will be searched
        path (string): '/' or '.' string with attribute names
    """
    return _get_element_by_names(source, re.split(separator, path))