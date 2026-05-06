def reference(node):
    """
    A hyperlink. Note that alt text doesn't work, since there's no apparent way to do that in docutils
    """
    o = nodes.reference()
    o['refuri'] = node.destination
    if node.title:
        o['name'] = node.title
    for n in MarkDown(node):
        o += n
    return o