def infer_type(expr, scope):
    """Try to infer the type of x.y if y is a known value (literal)."""
    # Do we know what the member is?
    if isinstance(expr.member, ast.Literal):
        member = expr.member.value
    else:
        return protocol.AnyType

    container_type = infer_type(expr.obj, scope)

    try:
        # We are not using lexical scope here on purpose - we want to see what
        # the type of the member is only on the container_type.
        return structured.reflect(container_type, member) or protocol.AnyType
    except NotImplementedError:
        return protocol.AnyType