def matchesExpr(cntxt: Context, T: RDFGraph, expr: ShExJ.tripleExpr, _: DebugContext) -> bool:
    """ Evaluate the expression

    """

    if isinstance(expr, ShExJ.OneOf):
        return matchesOneOf(cntxt, T, expr)
    elif isinstance(expr, ShExJ.EachOf):
        return matchesEachOf(cntxt, T, expr)
    elif isinstance(expr, ShExJ.TripleConstraint):
        return matchesCardinality(cntxt, T, expr)
    elif isinstance_(expr, ShExJ.tripleExprLabel):
        return matchesTripleExprRef(cntxt, T, expr)
    else:
        raise Exception("Unknown expression")