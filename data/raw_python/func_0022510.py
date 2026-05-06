def ReverseComplementMembership(x, y, **kwargs):
    """Change (x doesn't contain y) to not(y in x)."""
    return ast.Complement(
        ast.Membership(y, x, **kwargs), **kwargs)