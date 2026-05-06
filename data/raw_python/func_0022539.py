def infer_type(expr, scope):
    """Try to infer the type of x[y] if y is a known value (literal)."""
    # Do we know what the key even is?
    if isinstance(expr.key, ast.Literal):
        key = expr.key.value
    else:
        return protocol.AnyType

    container_type = infer_type(expr.value, scope)

    try:
        # Associative types are not subject to scoping rules so we can just
        # reflect using IAssociative.
        return associative.reflect(container_type, key) or protocol.AnyType
    except NotImplementedError:
        return protocol.AnyType