def predicates(G: Graph, n: Node) -> Set[TriplePredicate]:
    """ redicates(G, n) is the set of predicates in neigh(G, n).

        predicates(G, n) = predicatesOut(G, n) ∪ predicatesIn(G, n)
    """
    return predicatesOut(G, n) | predicatesIn(G, n)