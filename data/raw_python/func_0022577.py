def normalize(expr):
    """No elimination, but normalize arguments."""
    args = [normalize(arg) for arg in expr.args]

    return type(expr)(expr.func, *args, start=expr.start, end=expr.end)