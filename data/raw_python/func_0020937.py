def xpath(node, query, namespaces={}):
    """A safe xpath that only uses namespaces if available."""
    if namespaces and 'None' not in namespaces:
        return node.xpath(query, namespaces=namespaces)
    return node.xpath(query)