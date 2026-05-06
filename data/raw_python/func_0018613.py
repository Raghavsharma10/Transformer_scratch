def wrap_inner(node, tag):
    """Wrap the given tag around the contents of a node."""
    children = list(node.childNodes)
    wrap_node = node.ownerDocument.createElement(tag)
    for c in children:
        wrap_node.appendChild(c)
    node.appendChild(wrap_node)