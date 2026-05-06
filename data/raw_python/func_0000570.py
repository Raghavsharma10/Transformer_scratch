def filter_assert_nodes(nodes: List[ast.stmt], min_line_number: int) -> List[ast.stmt]:
    """
    Finds all nodes that are after the ``min_line_number``
    """
    return [node for node in nodes if node.lineno > min_line_number]