def satisifesShapeOr(cntxt: Context, n: Node, se: ShExJ.ShapeOr, _: DebugContext) -> bool:
    """ Se is a ShapeOr and there is some shape expression se2 in shapeExprs such that satisfies(n, se2, G, m). """
    return any(satisfies(cntxt, n, se2) for se2 in se.shapeExprs)