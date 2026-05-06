def base_url(root):
    """Determine the base url for a root element."""
    for attr, value in root.attrib.iteritems():
        if attr.endswith('base') and 'http' in value:
            return value
    return None