def dump_etree(data, container=None, nsmap=None, attribs=None):
    """Convert dictionary to Simple Dublin Core XML as ElementTree.

    :param data: Dictionary.
    :param container: Name (include namespace) of container element.
    :param nsmap: Namespace mapping for lxml.
    :param attribs: Default attributes for container element.
    :returns: LXML ElementTree.
    """
    container = container or container_element
    nsmap = nsmap or ns
    attribs = attribs or container_attribs
    return dump_etree_helper(container, data, rules, nsmap, attribs)