def find_stringy_lines(tree: ast.AST, first_line_no: int) -> Set[int]:
    """
    Finds all lines that contain a string in a tree, usually a function. These
    lines will be ignored when searching for blank lines.
    """
    str_footprints = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Str):
            str_footprints.update(build_footprint(node, first_line_no))
    return str_footprints