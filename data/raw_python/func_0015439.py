def matchesOneOf(cntxt: Context, T: RDFGraph, expr: ShExJ.OneOf, _: DebugContext) -> bool:
    """
    expr is a OneOf and there is some shape expression se2 in shapeExprs such that a matches(T, se2, m).
    """
    return any(matches(cntxt, T, e) for e in expr.expressions)