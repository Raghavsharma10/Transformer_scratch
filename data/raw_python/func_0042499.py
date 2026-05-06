def adjust_locations(ast_node, first_lineno, first_offset):
    """
    Adjust the locations of the ast nodes, offsetting them
    to the new lineno and column offset
    """

    line_delta = first_lineno - 1

    def _fix(node):
        if 'lineno' in node._attributes:
            lineno = node.lineno
            col = node.col_offset

            # adjust the offset on the first line
            if lineno == 1:
                col += first_offset

            lineno += line_delta

            node.lineno = lineno
            node.col_offset = col

        for child in iter_child_nodes(node):
            _fix(child)

    _fix(ast_node)