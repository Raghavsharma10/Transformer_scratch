def sparse_regularized_graph(X, positive=False, sparsity_param=None, kmax=None):
  '''Sparse Regularized Graph Construction, commonly known as an l1-graph.

  positive : bool, optional
    When True, computes the Sparse Probability Graph (SPG).
  sparsity_param : float, optional
    Controls sparsity cost in the LASSO optimization.
    When None, uses cross-validation to find sparsity parameters.
    This is very slow, but it gets good results.
  kmax : int, optional
    When None, allow all points to be edges. Otherwise, restrict to kNN set.

  l1-graph: "Semi-supervised Learning by Sparse Representation"
    Yan & Wang, SDM 2009
    http://epubs.siam.org/doi/pdf/10.1137/1.9781611972795.68

  SPG: "Nonnegative Sparse Coding for Discriminative Semi-supervised Learning"
    He et al., CVPR 2001
  '''
  clf, X = _l1_graph_setup(X, positive, sparsity_param)
  if kmax is None:
    W = _l1_graph_solve_full(clf, X)
  else:
    W = _l1_graph_solve_k(clf, X, kmax)
  return Graph.from_adj_matrix(W)