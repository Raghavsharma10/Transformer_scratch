def triple_in_shape(expr: ShExJ.shapeExpr, label: ShExJ.tripleExprLabel, cntxt: Context) \
        -> Optional[ShExJ.tripleExpr]:
    """ Search for the label in a shape expression """
    te = None
    if isinstance(expr, (ShExJ.ShapeOr, ShExJ.ShapeAnd)):
        for expr2 in expr.shapeExprs:
            te = triple_in_shape(expr2, label, cntxt)
            if te is not None:
                break
    elif isinstance(expr, ShExJ.ShapeNot):
        te = triple_in_shape(expr.shapeExpr, label, cntxt)
    elif isinstance(expr, ShExJ.shapeExprLabel):
        se = reference_of(expr, cntxt)
        if se is not None:
            te = triple_in_shape(se, label, cntxt)
    return te