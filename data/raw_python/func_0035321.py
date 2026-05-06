def connected_subgraphs(self, directed=True, ordered=False):
    '''Generates connected components as subgraphs.
    When ordered=True, subgraphs are ordered by number of vertices.
    '''
    num_ccs, labels = self.connected_components(directed=directed)
    # check the trivial case first
    if num_ccs == 1:
      yield self
      raise StopIteration
    if ordered:
      # sort by descending size (num vertices)
      order = np.argsort(np.bincount(labels))[::-1]
    else:
      order = range(num_ccs)

    # don't use self.subgraph() here, because we can reuse adj
    adj = self.matrix('dense', 'csr', 'csc')
    for c in order:
      mask = labels == c
      sub_adj = adj[mask][:,mask]
      yield self.__class__.from_adj_matrix(sub_adj)