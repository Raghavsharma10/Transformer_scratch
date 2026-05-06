def conforms(cntxt: Context, n: Node, S: ShExJ.Shape) -> bool:
    """ `5.6.1 Schema Validation Requirement <http://shex.io/shex-semantics/#validation-requirement>`_
    
    A graph G is said to conform with a schema S with a ShapeMap m when:

    Every, SemAct in the startActs of S has a successful evaluation of semActsSatisfied.
    Every node n in m conforms to its associated shapeExprRefs sen where for each shapeExprRef sei in sen:
        sei references a ShapeExpr in shapes, and
        satisfies(n, sei, G, m) for each shape sei in sen.

    :return:
    """
    # return semActsSatisfied(cntxt.schema.startActs, cntxt) and \
    #     all(reference_of(cntxt.schema, sa.shapeLabel) is not None and
    #
    return True