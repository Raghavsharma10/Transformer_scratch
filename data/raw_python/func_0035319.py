def kernelize(self, kernel):
    '''Re-weight according to a specified kernel function.
    kernel : str, {none, binary, rbf}
      none   -> no reweighting
      binary -> all edges are given weight 1
      rbf    -> applies a gaussian function to edge weights
    '''
    if kernel == 'none':
      return self
    if kernel == 'binary':
      if self.is_weighted():
        return self._update_edges(1, copy=True)
      return self
    if kernel == 'rbf':
      w = self.edge_weights()
      r = np.exp(-w / w.std())
      return self._update_edges(r, copy=True)
    raise ValueError('Invalid kernel type: %r' % kernel)