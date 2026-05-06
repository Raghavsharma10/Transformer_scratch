def reweight(self, weight, edges=None, copy=False):
    '''Replaces existing edge weights. weight may be a scalar or 1d array.
    edges is a mask or index array that specifies a subset of edges to modify'''
    if not self.is_weighted():
      warnings.warn('Cannot supply weights for unweighted graph; '
                    'ignoring call to reweight')
      return self
    if edges is None:
      return self._update_edges(weight, copy=copy)
    ii, jj = self.pairs()[edges].T
    return self.add_edges(ii, jj, weight=weight, symmetric=False, copy=copy)