def satisfiesShapeExprRef(cntxt: Context, n: Node, se: ShExJ.shapeExprLabel, c: DebugContext) -> bool:
    """ Se is a shapeExprRef and there exists in the schema a shape expression se2 with that id
     and satisfies(n, se2, G, m).
     """
    if c.debug:
        print(f"id: {se}")
    for shape in cntxt.schema.shapes:
        if shape.id == se:
            return satisfies(cntxt, n, shape)
    cntxt.fail_reason = f"{se}: Shape is not in Schema"
    return False