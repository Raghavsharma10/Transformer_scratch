def satisfies(cntxt: Context, n: Node, se: ShExJ.shapeExpr) -> bool:
    """ `5.3 Shape Expressions <http://shex.io/shex-semantics/#node-constraint-semantics>`_

          satisfies: The expression satisfies(n, se, G, m) indicates that a node n and graph G satisfy a shape
                      expression se with shapeMap m.

           satisfies(n, se, G, m) is true if and only if:

            * Se is a NodeConstraint and satisfies2(n, se) as described below in Node Constraints.
                      Note that testing if a node satisfies a node constraint does not require a graph or shapeMap.
            * Se is a Shape and satisfies(n, se) as defined below in Shapes and Triple Expressions.
            * Se is a ShapeOr and there is some shape expression se2 in shapeExprs such that
            satisfies(n, se2, G, m).
            * Se is a ShapeAnd and for every shape expression se2 in shapeExprs, satisfies(n, se2, G, m).
            * Se is a ShapeNot and for the shape expression se2 at shapeExpr, notSatisfies(n, se2, G, m).
            * Se is a ShapeExternal and implementation-specific mechansims not defined in this specification
            indicate success.
            * Se is a shapeExprRef and there exists in the schema a shape expression se2 with that id and
                      satisfies(n, se2, G, m).

          .. note:: Where is the documentation on recursion?  All I can find is
           `5.9.4 Recursion Example <http://shex.io/shex-semantics/#example-recursion>`_
          """
    if isinstance(se, ShExJ.NodeConstraint):
        rval = satisfiesNodeConstraint(cntxt, n, se)
    elif isinstance(se, ShExJ.Shape):
        rval = satisfiesShape(cntxt, n, se)
    elif isinstance(se, ShExJ.ShapeOr):
        rval = satisifesShapeOr(cntxt, n, se)
    elif isinstance(se, ShExJ.ShapeAnd):
        rval = satisfiesShapeAnd(cntxt, n, se)
    elif isinstance(se, ShExJ.ShapeNot):
        rval = satisfiesShapeNot(cntxt, n, se)
    elif isinstance(se, ShExJ.ShapeExternal):
        rval = satisfiesExternal(cntxt, n, se)
    elif isinstance_(se, ShExJ.shapeExprLabel):
        rval = satisfiesShapeExprRef(cntxt, n, se)
    else:
        raise NotImplementedError(f"Unrecognized shapeExpr: {type(se)}")
    return rval