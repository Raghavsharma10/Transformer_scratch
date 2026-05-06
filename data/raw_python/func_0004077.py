def visit(d, op):
    """Recursively call op(d) for all list subelements and dictionary 'values' that d may have."""
    op(d)
    if isinstance(d, list):
        for i in d:
            visit(i, op)
    elif isinstance(d, dict):
        for i in itervalues(d):
            visit(i, op)