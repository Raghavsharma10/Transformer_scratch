def section(node):
    """
    A section in reStructuredText, which needs a title (the first child)
    This is a custom type
    """
    title = ''  # All sections need an id
    if node.first_child is not None:
        if node.first_child.t == u'heading':
            title = node.first_child.first_child.literal
    o = nodes.section(ids=[title], names=[title])
    for n in MarkDown(node):
        o += n
    return o