def emphasis(node):
    """
    An italicized section
    """
    o = nodes.emphasis()
    for n in MarkDown(node):
        o += n
    return o