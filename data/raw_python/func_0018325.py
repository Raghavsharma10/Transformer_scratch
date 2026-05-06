def strong(node):
    """
    A bolded section
    """
    o = nodes.strong()
    for n in MarkDown(node):
        o += n
    return o