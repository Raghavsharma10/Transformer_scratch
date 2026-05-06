def neighbor_graph(X, metric='euclidean', k=None, epsilon=None,
                   weighting='none', precomputed=False):
  '''Build a neighbor graph from pairwise distance information.

  X : two-dimensional array-like
      Shape must either be (num_pts, num_dims) or (num_pts, num_pts).
  k : int, maximum number of nearest neighbors
  epsilon : float, maximum distance to a neighbor
  metric : str, type of distance metric (see sklearn.metrics)
      When metric='precomputed', X is a symmetric distance matrix.
  weighting : str, one of {'binary', 'none'}
      When weighting='binary', all edge weights == 1.
  '''
  if k is None and epsilon is None:
    raise ValueError('Must provide `k` or `epsilon`.')
  if weighting not in ('binary', 'none'):
    raise ValueError('Invalid weighting param: %r' % weighting)

  # TODO: deprecate the precomputed kwarg
  precomputed = precomputed or (metric == 'precomputed')
  binary = weighting == 'binary'

  # Try the fast path, if possible.
  if not precomputed and epsilon is None:
    return _sparse_neighbor_graph(X, k, binary, metric)

  if precomputed:
    D = X
  else:
    D = pairwise_distances(X, metric=metric)
  return _slow_neighbor_graph(D, k, epsilon, binary)