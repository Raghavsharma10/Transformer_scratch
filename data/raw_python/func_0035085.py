def to_networkx(self, directed=None):
    '''Converts this Graph object to a networkx-compatible object.
    Requires the networkx library.'''
    import networkx as nx
    directed = directed if directed is not None else self.is_directed()
    cls = nx.DiGraph if directed else nx.Graph
    adj = self.matrix()
    if ss.issparse(adj):
      return nx.from_scipy_sparse_matrix(adj, create_using=cls())
    return nx.from_numpy_matrix(adj, create_using=cls())