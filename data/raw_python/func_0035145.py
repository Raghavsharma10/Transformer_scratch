def _sparse_neighbor_graph(X, k, binary=False, metric='l2'):
  '''Construct a sparse adj matrix from a matrix of points (one per row).
  Non-zeros are unweighted/binary distance values, depending on the binary arg.
  Doesn't include self-edges.'''
  knn = NearestNeighbors(n_neighbors=k, metric=metric).fit(X)
  mode = 'connectivity' if binary else 'distance'
  try:
    adj = knn.kneighbors_graph(None, mode=mode)
  except IndexError:
    # XXX: we must be running an old (<0.16) version of sklearn
    #  We have to hack around an old bug:
    if binary:
      adj = knn.kneighbors_graph(X, k+1, mode=mode)
      adj.setdiag(0)
    else:
      adj = knn.kneighbors_graph(X, k, mode=mode)
  return Graph.from_adj_matrix(adj)