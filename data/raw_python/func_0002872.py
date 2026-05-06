def comment(node):
    """
    Converts the node received to a comment, in place, and will also return the
    comment element.
    """
    parent = node.parentNode
    comment = node.ownerDocument.createComment(node.toxml())
    parent.replaceChild(comment, node)
    return comment