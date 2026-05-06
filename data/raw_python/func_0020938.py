def innertext(node):
    """Return the inner text of a node.  If a node has no sub elements, this
    is just node.text.  Otherwise, it's node.text + sub-element-text +
    node.tail."""
    if not len(node):
        return node.text

    return (node.text or '') + ''.join([etree.tostring(c) for c in node]) + (node.tail or '')