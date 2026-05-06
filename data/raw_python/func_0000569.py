def filter_arrange_nodes(nodes: List[ast.stmt], max_line_number: int) -> List[ast.stmt]:
    """
    Finds all nodes that are before the ``max_line_number`` and are not
    docstrings or ``pass``.
    """
    return [
        node for node in nodes if node.lineno < max_line_number and not isinstance(node, ast.Pass)
        and not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Str))
    ]