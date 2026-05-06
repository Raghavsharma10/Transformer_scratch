def satisfiesNodeConstraint(cntxt: Context, n: Node, nc: ShExJ.NodeConstraint, _: DebugContext) -> bool:
    """ `5.4.1 Semantics <http://shex.io/shex-semantics/#node-constraint-semantics>`_

    For a node n and constraint nc, satisfies2(n, nc) if and only if for every nodeKind, datatype, xsFacet and
    values constraint value v present in nc nodeSatisfies(n, v). The following sections define nodeSatisfies for
    each of these types of constraints:
    """
    return nodeSatisfiesNodeKind(cntxt, n, nc) and nodeSatisfiesDataType(cntxt, n, nc) and \
        nodeSatisfiesStringFacet(cntxt, n, nc) and nodeSatisfiesNumericFacet(cntxt, n, nc) and \
        nodeSatisfiesValues(cntxt, n, nc)