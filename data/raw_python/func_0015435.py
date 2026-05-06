def valid_remainder(cntxt: Context, n: Node, matchables: RDFGraph, S: ShExJ.Shape) -> bool:
    """
    Let **outs** be the arcsOut in remainder: `outs = remainder ∩ arcsOut(G, n)`.

    Let **matchables** be the triples in outs whose predicate appears in a TripleConstraint in `expression`. If
    `expression` is absent, matchables = Ø (the empty set).

    * There is no triple in **matchables** which matches a TripleConstraint in expression

    * There is no triple in **matchables** whose predicate does not appear in extra.

    * closed is false or unmatchables is empty

    :param cntxt: evaluation context
    :param n: focus node
    :param matchables: non-matched triples
    :param S: Shape being evaluated
    :return: True if remainder is valid
    """
    # TODO: Update this and satisfies to address the new algorithm
    # Let **outs** be the arcsOut in remainder: `outs = remainder ∩ arcsOut(G, n)`.
    outs = arcsOut(cntxt.graph, n).intersection(matchables)

    # predicates that in a TripleConstraint in `expression`
    predicates = predicates_in_expression(S, cntxt)

    # Let **matchables** be the triples in outs whose predicate appears in predicates. If
    # `expression` is absent, matchables = Ø (the empty set).
    matchables = RDFGraph(t for t in outs if str(t.p) in predicates)

    # There is no triple in **matchables** which matches a TripleConstraint in expression
    if matchables and S.expression is not None:
        tes = triple_constraints_in_expression(S.expression, cntxt)
        for m in matchables:
            if any(matchesTripleConstraint(cntxt, m, te) for te in tes):
                return False

    # There is no triple in **matchables** whose predicate does not appear in extra.
    extras = {iriref_to_uriref(e) for e in S.extra} if S.extra is not None else {}
    if any(t.p not in extras for t in matchables):
        return False

    # closed is false or unmatchables is empty.
    return not S.closed.val or not bool(outs - matchables)