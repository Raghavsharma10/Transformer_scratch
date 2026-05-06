def predicates_in_expression(expression: ShExJ.shapeExpr, cntxt: Context) -> List[IRIREF]:
    """ Return the set of predicates that "appears in a TripleConstraint in an expression
    
    See: `5.5.2 Semantics <http://shex.io/shex-semantics/#triple-expressions-semantics>`_ for details

    :param expression: Expression to scan for predicates
    :param cntxt: Context of evaluation
    :return: List of predicates
    """
    return list(directed_predicates_in_expression(expression, cntxt).keys())