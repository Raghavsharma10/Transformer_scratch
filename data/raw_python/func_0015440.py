def matchesEachOf(cntxt: Context, T: RDFGraph, expr: ShExJ.EachOf, _: DebugContext) -> bool:
    """ expr is an EachOf and there is some partition of T into T1, T2,… such that for every expression
     expr1, expr2,… in shapeExprs, matches(Tn, exprn, m).
     """

    return EachOfEvaluator(cntxt, T, expr).evaluate(cntxt)