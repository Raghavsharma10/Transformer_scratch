def is_loop_helper(node):
    """
    Returns True is node is a loop helper e.g. {{ loop.index }} or {{ loop.first }}
    """
    return hasattr(node, 'node') and isinstance(node.node, nodes.Name) and node.node.name == 'loop'