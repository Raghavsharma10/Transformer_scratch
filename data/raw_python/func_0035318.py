def disjoint_mst(X, num_spanning_trees=3, metric='euclidean'):
  '''Builds a graph as the union of several spanning trees,
  each time removing any edges present in previously-built trees.
  Reference: http://ecovision.mit.edu/~sloop/shao.pdf, page 9.'''
  D = pairwise_distances(X, metric=metric)
  if metric == 'precomputed':
    D = D.copy()
  mst = minimum_spanning_tree(D)
  W = mst.copy()
  for i in range(1, num_spanning_trees):
    ii,jj = mst.nonzero()
    D[ii,jj] = np.inf
    D[jj,ii] = np.inf
    mst = minimum_spanning_tree(D)
    W = W + mst
  # MSTs are all one-sided, so we symmetrize here
  return Graph.from_adj_matrix(W + W.T)