def add_node_parents(root: ast.AST) -> None:
    """
    Adds "parent" attribute to all child nodes of passed node.

    Code taken from https://stackoverflow.com/a/43311383/1286705
    """
    for node in ast.walk(root):
        for child in ast.iter_child_nodes(node):
            child.parent = node