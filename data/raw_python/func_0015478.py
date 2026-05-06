def satisfiesShapeAnd(cntxt: Context, n: Node, se: ShExJ.ShapeAnd, _: DebugContext) -> bool:
    """ Se is a ShapeAnd and for every shape expression se2 in shapeExprs, satisfies(n, se2, G, m) """
    return all(satisfies(cntxt, n, se2) for se2 in se.shapeExprs)