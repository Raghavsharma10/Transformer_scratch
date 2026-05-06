def triple_reference_of(label: ShExJ.tripleExprLabel, cntxt: Context) -> Optional[ShExJ.tripleExpr]:
    """ Search for the label in a Schema """
    te: Optional[ShExJ.tripleExpr] = None
    if cntxt.schema.start is not None:
        te = triple_in_shape(cntxt.schema.start, label, cntxt)
    if te is None:
        for shapeExpr in cntxt.schema.shapes:
            te = triple_in_shape(shapeExpr, label, cntxt)
            if te:
                break
    return te