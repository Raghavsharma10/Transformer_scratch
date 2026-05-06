def reference_of(selector: shapeLabel, cntxt: Union[Context, ShExJ.Schema] ) -> Optional[ShExJ.shapeExpr]:
    """ Return the shape expression in the schema referenced by selector, if any

    :param cntxt: Context node or ShEx Schema
    :param selector: identifier of element to select within the schema
    :return:
    """
    schema = cntxt.schema if isinstance(cntxt, Context) else cntxt
    if selector is START:
        return schema.start
    for expr in schema.shapes:
        if not isinstance(expr, ShExJ.ShapeExternal) and expr.id == selector:
            return expr
    return schema.start if schema.start is not None and schema.start.id == selector else None