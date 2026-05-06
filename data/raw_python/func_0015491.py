def predicatesOut(G: Graph, n: Node) -> Set[TriplePredicate]:
    """ predicatesOut(G, n) is the set of predicates in arcsOut(G, n). """
    return {p for p, _ in G.predicate_objects(n)}