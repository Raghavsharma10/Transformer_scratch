def arcsOut(G: Graph, n: Node) -> RDFGraph:
    """ arcsOut(G, n) is the set of triples in a graph G with subject n. """
    return RDFGraph(G.triples((n, None, None)))