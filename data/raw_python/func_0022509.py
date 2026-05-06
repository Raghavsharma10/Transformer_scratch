def ComplementMembership(*args, **kwargs):
    """Change (x not in y) to not(x in y)."""
    return ast.Complement(
        ast.Membership(*args, **kwargs), **kwargs)