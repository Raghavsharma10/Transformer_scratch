def directed_predicates_in_expression(expression: ShExJ.shapeExpr, cntxt: Context) -> Dict[IRIREF, PredDirection]:
    """ Directed predicates in expression -- return all predicates in shapeExpr along with which direction(s) they
    evaluate

    :param expression: Expression to scan
    :param cntxt:
    :return:
    """
    dir_predicates: Dict[IRIREF, PredDirection] = {}

    def predicate_finder(predicates: Dict[IRIREF, PredDirection], tc: ShExJ.TripleConstraint, _: Context) -> None:
        if isinstance(tc, ShExJ.TripleConstraint):
            predicates.setdefault(tc.predicate, PredDirection()).dir(tc.inverse is None or not tc.inverse)

    def triple_expr_finder(predicates: Dict[IRIREF, PredDirection], expr: ShExJ.shapeExpr, cntxt_: Context) -> None:
        if isinstance(expr, ShExJ.Shape) and expr.expression is not None:
            cntxt_.visit_triple_expressions(expr.expression, predicate_finder, predicates)

    # TODO: follow_inner_shapes as True probably goes too far, but we definitely need to cross shape/triplecons
    cntxt.visit_shapes(expression, triple_expr_finder, dir_predicates, follow_inner_shapes=False)
    return dir_predicates