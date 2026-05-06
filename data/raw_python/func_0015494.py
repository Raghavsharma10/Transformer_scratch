def neigh(G: Graph, n: Node) -> RDFGraph:
    """  neigh(G, n) is the neighbourhood of the node n in the graph G.

         neigh(G, n) = arcsOut(G, n) ∪ arcsIn(G, n)
    """
    return arcsOut(G, n) | arcsIn(G, n)