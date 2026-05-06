def matchesTripleConstraint(cntxt: Context, t: RDFTriple, expr: ShExJ.TripleConstraint, c: DebugContext) -> bool:
    """
    expr is a TripleConstraint and:

    * t is a triple
    * t's predicate equals expr's predicate.
      Let value be t's subject if inverse is true, else t's object.
    * if inverse is true, t is in arcsIn, else t is in arcsOut.

    """
    from pyshex.shape_expressions_language.p5_3_shape_expressions import satisfies

    if c.debug:
        print(c.i(1, f" triple: {t}"))
        print(c.i(1, '', expr._as_json_dumps().split('\n')))

    if uriref_matches_iriref(t.p, expr.predicate):
        value = t.s if expr.inverse else t.o
        return expr.valueExpr is None or satisfies(cntxt, value, expr.valueExpr)
    else:
        cntxt.fail_reason = f"Predicate mismatch: {t.p} ≠ {expr.predicate}"
        return False