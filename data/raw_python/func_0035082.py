def degree(self, kind='out', weighted=True):
    '''Returns an array of vertex degrees.
    kind : either 'in' or 'out', useful for directed graphs
    weighted : controls whether to count edges or sum their weights
    '''
    if kind == 'out':
      axis = 1
      adj = self.matrix('dense', 'csc')
    else:
      axis = 0
      adj = self.matrix('dense', 'csr')

    if not weighted and self.is_weighted():
      # With recent numpy and a dense matrix, could do:
      # d = np.count_nonzero(adj, axis=axis)
      d = (adj!=0).sum(axis=axis)
    else:
      d = adj.sum(axis=axis)
    return np.asarray(d).ravel()