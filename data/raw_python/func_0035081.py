def reweight_by_distance(self, coords, metric='l2', copy=False):
    '''Replaces existing edge weights by distances between connected vertices.
    The new weight of edge (i,j) is given by: metric(coords[i], coords[j]).
    coords : (num_vertices x d) array of coordinates, in vertex order
    metric : str or callable, see sklearn.metrics.pairwise.paired_distances'''
    if not self.is_weighted():
      warnings.warn('Cannot supply weights for unweighted graph; '
                    'ignoring call to reweight_by_distance')
      return self
    # TODO: take advantage of symmetry of metric function
    ii, jj = self.pairs().T
    if metric == 'precomputed':
      assert coords.ndim == 2 and coords.shape[0] == coords.shape[1]
      d = coords[ii,jj]
    else:
      d = paired_distances(coords[ii], coords[jj], metric=metric)
    return self._update_edges(d, copy=copy)