def satisfiesShapeNot(cntxt: Context, n: Node, se: ShExJ.ShapeNot, _: DebugContext) -> bool:
    """ Se is a ShapeNot and for the shape expression se2 at shapeExpr, notSatisfies(n, se2, G, m) """
    return not satisfies(cntxt, n, se.shapeExpr)