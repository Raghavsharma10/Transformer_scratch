def satisfiesShape(cntxt: Context, n: Node, S: ShExJ.Shape, c: DebugContext) -> bool:
    """ `5.5.2 Semantics <http://shex.io/shex-semantics/#triple-expressions-semantics>`_

    For a node `n`, shape `S`, graph `G`, and shapeMap `m`, `satisfies(n, S, G, m)` if and only if:

    * `neigh(G, n)` can be partitioned into two sets matched and remainder such that
      `matches(matched, expression, m)`. If expression is absent, remainder = `neigh(G, n)`.

    :param n: focus node
    :param S: Shape to be satisfied
    :param cntxt: Evaluation context
    :param c: Debug context
    :return: true iff `satisfies(n, S, cntxt)`
    """

    # Recursion detection.  If start_evaluating returns a boolean value, this is the assumed result of the shape
    # evaluation.  If it returns None, then an initial evaluation is needed
    rslt = cntxt.start_evaluating(n, S)

    if rslt is None:
        cntxt.evaluate_stack.append((n, S.id))
        predicates = directed_predicates_in_expression(S, cntxt)
        matchables = RDFGraph()

        # Note: The code below does an "over-slurp" for the sake of expediency.  If you are interested in
        #       getting EXACTLY the needed triples, set cntxt.over_slurp to false
        if isinstance(cntxt.graph, SlurpyGraph) and cntxt.over_slurp:
            with slurper(cntxt, n, S) as g:
                _ = g.triples((n, None, None))

        for predicate, direction in predicates.items():
            with slurper(cntxt, n, S) as g:
                matchables.add_triples(g.triples((n if direction.is_fwd else None,
                                                  iriref_to_uriref(predicate),
                                                  n if direction.is_rev else None)))

        if c.debug:
            print(c.i(1, "predicates:", sorted(cntxt.n3_mapper.n3(p) for p in predicates.keys())))
            print(c.i(1, "matchables:", sorted(cntxt.n3_mapper.n3(m) for m in matchables)))
            print()

        if S.closed:
            # TODO: Is this working correctly on reverse items?
            non_matchables = RDFGraph([t for t in arcsOut(cntxt.graph, n) if t not in matchables])
            if len(non_matchables):
                cntxt.fail_reason = "Unmatched triples in CLOSED shape:"
                cntxt.fail_reason = '\n'.join(f"\t{t}" for t in non_matchables)
                if c.debug:
                    print(c.i(0,
                              f"<--- Satisfies shape {c.d()} FAIL - "
                              f"{len(non_matchables)} non-matching triples on a closed shape"))
                    print(c.i(1, "", list(non_matchables)))
                    print()
                return False


        # Evaluate the actual expression.  Start assuming everything matches...
        if S.expression:
            if matches(cntxt, matchables, S.expression):
                rslt = True
            else:
                extras = {iriref_to_uriref(e) for e in S.extra} if S.extra is not None else {}
                if len(extras):
                    permutable_matchables = RDFGraph([t for t in matchables if t.p in extras])
                    non_permutable_matchables = RDFGraph([t for t in matchables if t not in permutable_matchables])
                    if c.debug:
                        print(c.i(1,
                                  f"Complete match failed -- evaluating extras", list(extras)))
                    for matched, remainder in partition_2(permutable_matchables):
                        permutation = non_permutable_matchables.union(matched)
                        if matches(cntxt, permutation, S.expression):
                            rslt = True
                            break
                rslt = rslt or False
        else:
            rslt = True         # Empty shape

        # If an assumption was made and the result doesn't match the assumption, switch directions and try again
        done, consistent = cntxt.done_evaluating(n, S, rslt)
        if not done:
            rslt = satisfiesShape(cntxt, n, S)
        rslt = rslt and consistent

        cntxt.evaluate_stack.pop()
    return rslt