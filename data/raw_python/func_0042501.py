def remove_locations(node):
    """
    Removes locations from the given AST tree completely
    """

    def fix(node):
        if 'lineno' in node._attributes and hasattr(node, 'lineno'):
            del node.lineno

        if 'col_offset' in node._attributes and hasattr(node, 'col_offset'):
            del node.col_offset

        for child in iter_child_nodes(node):
            fix(child)

    fix(node)