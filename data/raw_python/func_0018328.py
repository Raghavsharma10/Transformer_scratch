def title(node):
    """
    A title node. It has no children
    """
    return nodes.title(node.first_child.literal, node.first_child.literal)