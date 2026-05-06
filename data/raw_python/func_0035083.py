def to_igraph(self, weighted=None):
    '''Converts this Graph object to an igraph-compatible object.
    Requires the python-igraph library.'''
    # Import here to avoid ImportErrors when igraph isn't available.
    import igraph
    ig = igraph.Graph(n=self.num_vertices(), edges=self.pairs().tolist(),
                      directed=self.is_directed())
    if weighted is not False and self.is_weighted():
      ig.es['weight'] = self.edge_weights()
    return ig