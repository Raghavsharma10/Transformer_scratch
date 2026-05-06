def extract_nodes(soup, nodename, attr = None, value = None):
    """
    Returns a list of tags (nodes) from the given soup matching the given nodename.
    If an optional attribute and value are given, these are used to filter the results
    further."""
    tags = soup.find_all(nodename)
    if attr != None and value != None:
        return list(filter(lambda tag: tag.get(attr) == value, tags))
    return list(tags)