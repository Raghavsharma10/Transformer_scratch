def predicatesIn(G: Graph, n: Node) -> Set[TriplePredicate]:
    """ predicatesIn(G, n) is the set of predicates in arcsIn(G, n). """
    return {p for _, p in G.subject_predicates(n)}