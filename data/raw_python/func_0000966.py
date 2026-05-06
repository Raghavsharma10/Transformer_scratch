def _get_simple_assignments(tree):
    """Get simple assignments from node tree."""
    result = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    result[target.id] = node.value
    return result