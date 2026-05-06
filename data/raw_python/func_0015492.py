def arcsIn(G: Graph, n: Node) -> RDFGraph:
    """ arcsIn(G, n) is the set of triples in a graph G with object n. """
    return RDFGraph(G.triples((None, None, n)))