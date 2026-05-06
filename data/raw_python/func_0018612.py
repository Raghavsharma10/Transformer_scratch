def wrap(node, tag):
    """Wrap the given tag around a node."""
    wrap_node = node.ownerDocument.createElement(tag)
    parent = node.parentNode
    if parent:
        parent.replaceChild(wrap_node, node)
    wrap_node.appendChild(node)
    return wrap_node