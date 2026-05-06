def getvar(syntree, targetvar):
    """Scan an ast object for targetvar and return its value.

    Only handles single direct assignment of python literal types. See docs on
    ast.literal_eval for more info:
    http://docs.python.org/2/library/ast.html#ast.literal_eval

    Args:
      syntree: ast.Module object
      targetvar: name of global variable to return
    Returns:
      Value of targetvar if found in syntree, or None if not found.
    """
    for node in syntree.body:
        if isinstance(node, ast.Assign):
            for var in node.targets:
                if var.id == targetvar:
                    return ast.literal_eval(node.value)