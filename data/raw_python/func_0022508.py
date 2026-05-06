def ComplementEquivalence(*args, **kwargs):
    """Change x != y to not(x == y)."""
    return ast.Complement(
        ast.Equivalence(*args, **kwargs), **kwargs)