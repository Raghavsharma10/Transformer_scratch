def smce_graph(X, metric='l2', sparsity_param=10, kmax=None, keep_ratio=0.95):
  '''Sparse graph construction from the SMCE paper.

  X : 2-dimensional array-like
  metric : str, optional
  sparsity_param : float, optional
  kmax : int, optional
  keep_ratio : float, optional
    When <1, keep edges up to (keep_ratio * total weight)

  Returns a graph with asymmetric similarity weights.
  Call .symmetrize() and .kernelize('rbf') to convert to symmetric distances.

  SMCE: "Sparse Manifold Clustering and Embedding"
    Elhamifar & Vidal, NIPS 2011
  '''
  n = X.shape[0]
  if kmax is None:
    kmax = min(n-1, max(5, n // 10))

  nn_dists, nn_inds = nearest_neighbors(X, metric=metric, k=kmax+1,
                                        return_dists=True)
  W = np.zeros((n, n))

  # optimize each point separately
  for i, pt in enumerate(X):
    nbr_inds = nn_inds[i]
    mask = nbr_inds != i  # remove self-edge
    nbr_inds = nbr_inds[mask]
    nbr_dist = nn_dists[i,mask]
    Y = (X[nbr_inds] - pt) / nbr_dist[:,None]
    # solve sparse optimization with ADMM
    c = _solve_admm(Y, nbr_dist/nbr_dist.sum(), sparsity_param)
    c = np.abs(c / nbr_dist)
    W[i,nbr_inds] = c / c.sum()

  W = ss.csr_matrix(W)
  if keep_ratio < 1:
    for i in range(n):
      row_data = W.data[W.indptr[i]:W.indptr[i+1]]
      order = np.argsort(row_data)[::-1]
      stop_idx = np.searchsorted(np.cumsum(row_data[order]), keep_ratio) + 1
      bad_inds = order[stop_idx:]
      row_data[bad_inds] = 0
    W.eliminate_zeros()

  return Graph.from_adj_matrix(W)