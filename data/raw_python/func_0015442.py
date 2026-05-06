def matchesTripleExprRef(cntxt: Context, T: RDFGraph, expr: ShExJ.tripleExprLabel, _: DebugContext) -> bool:
    """
    expr is an tripleExprRef and satisfies(value, tripleExprWithId(tripleExprRef), G, m).
    The tripleExprWithId function is defined in Triple Expression Reference Requirement below.
    """
    expr = cntxt.tripleExprFor(expr)
    if expr is None:
        cntxt.fail_reason = "{expr}: Reference not found"
        return False
    return matchesExpr(cntxt, T, expr)