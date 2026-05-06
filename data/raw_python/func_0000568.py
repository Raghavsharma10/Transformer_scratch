def build_footprint(node: ast.AST, first_line_no: int) -> Set[int]:
    """
    Generates a list of lines that the passed node covers, relative to the
    marked lines list - i.e. start of function is line 0.
    """
    return set(
        range(
            get_first_token(node).start[0] - first_line_no,
            get_last_token(node).end[0] - first_line_no + 1,
        )
    )