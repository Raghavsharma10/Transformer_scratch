def block_quote(node):
    """
    A block quote
    """
    o = nodes.block_quote()
    o.line = node.sourcepos[0][0]
    for n in MarkDown(node):
        o += n
    return o