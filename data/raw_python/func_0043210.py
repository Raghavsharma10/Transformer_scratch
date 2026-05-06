def get_compound_bodies(node):
    """Returns a list of bodies of a compound statement node.

    Args:
        node: AST node.

    Returns:
        A list of bodies of the node. If the given node does not represent
        a compound statement, an empty list is returned.
    """
    if isinstance(node, (ast.Module, ast.FunctionDef, ast.ClassDef, ast.With)):
        return [node.body]
    elif isinstance(node, (ast.If, ast.While, ast.For)):
        return [node.body, node.orelse]
    elif PY2 and isinstance(node, ast.TryFinally):
        return [node.body, node.finalbody]
    elif PY2 and isinstance(node, ast.TryExcept):
        return [node.body, node.orelse] + [h.body for h in node.handlers]
    elif PY3 and isinstance(node, ast.Try):
        return ([node.body, node.orelse, node.finalbody]
                + [h.body for h in node.handlers])
    end
    return []