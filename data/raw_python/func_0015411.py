def nodeSatisfiesNodeKind(cntxt: Context, n: Node, nc: ShExJ.NodeConstraint, c: DebugContext) -> bool:
    """ `5.4.2 Node Kind Constraints <http://shex.io/shex-semantics/#nodeKind>`_

    For a node n and constraint value v, nodeSatisfies(n, v) if:

        * v = "iri" and n is an IRI.
        * v = "bnode" and n is a blank node.
        * v = "literal" and n is a Literal.
        * v = "nonliteral" and n is an IRI or blank node.
    """
    if c.debug and nc.nodeKind is not None:
        print(f" Kind: {nc.nodeKind}")
    if nc.nodeKind is None or \
        (nc.nodeKind == 'iri' and isinstance(n, URIRef)) or \
        (nc.nodeKind == 'bnode' and isinstance(n, BNode)) or \
        (nc.nodeKind == 'literal' and isinstance(n, Literal)) or \
        (nc.nodeKind == 'nonliteral' and isinstance(n, (URIRef, BNode))):
        return True
    cntxt.fail_reason = f"Node kind mismatch have: {type(n).__name__} expected: {nc.nodeKind}"
    return False