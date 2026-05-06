def image(node):
    """
    An image element

    The first child is the alt text. reStructuredText can't handle titles
    """
    o = nodes.image(uri=node.destination)
    if node.first_child is not None:
        o['alt'] = node.first_child.literal
    return o