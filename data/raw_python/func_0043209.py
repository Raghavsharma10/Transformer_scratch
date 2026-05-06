def is_end_node(node):
    """Checks if a node is the "end" keyword.

    Args:
        node: AST node.

    Returns:
        True if the node is the "end" keyword, otherwise False.
    """
    return (isinstance(node, ast.Expr) and
            isinstance(node.value, ast.Name) and
            node.value.id == 'end')