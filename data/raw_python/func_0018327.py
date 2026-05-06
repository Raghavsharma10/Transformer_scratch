def raw(node):
    """
    Add some raw html (possibly as a block)
    """
    o = nodes.raw(node.literal, node.literal, format='html')
    if node.sourcepos is not None:
        o.line = node.sourcepos[0][0]
    for n in MarkDown(node):
        o += n
    return o