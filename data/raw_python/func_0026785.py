def node_is_embedded_doc_attr(node):
    """Checks if a node is a valid field or method in a embedded document.
    """
    embedded_doc = get_field_embedded_doc(node.last_child())
    name = node.attrname
    try:
        r = bool(embedded_doc.lookup(name)[1][0])
    except IndexError:
        r = False

    return r