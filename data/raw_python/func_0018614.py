def unwrap(node):
    """Remove a node, replacing it with its children."""
    for child in list(node.childNodes):
        node.parentNode.insertBefore(child, node)
    remove_node(node)