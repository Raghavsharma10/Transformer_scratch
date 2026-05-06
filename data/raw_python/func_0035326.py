def cycle_cut(self, cycle_len_thresh=12, directed=False, copy=True):
    '''CycleCut algorithm: removes bottleneck edges.
    Paper DOI: 10.1.1.225.5335
    '''
    symmetric = not directed
    adj = self.kernelize('binary').matrix('csr', 'dense', copy=True)
    if symmetric:
      adj = adj + adj.T

    removed_edges = []
    while True:
      c = _atomic_cycle(adj, cycle_len_thresh, directed=directed)
      if c is None:
        break
      # remove edges in the cycle
      ii, jj = c.T
      adj[ii,jj] = 0
      if symmetric:
        adj[jj,ii] = 0
      removed_edges.extend(c)

    #XXX: if _atomic_cycle changes, may need to do this on each loop
    if ss.issparse(adj):
      adj.eliminate_zeros()

    # select only the necessary cuts
    ii, jj = _find_cycle_inducers(adj, removed_edges, cycle_len_thresh,
                                  directed=directed)
    # remove the bad edges
    return self.remove_edges(ii, jj, symmetric=symmetric, copy=copy)