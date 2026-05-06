def listItem(node):
    """
    An item in a list
    """
    o = nodes.list_item()
    for n in MarkDown(node):
        o += n
    return o