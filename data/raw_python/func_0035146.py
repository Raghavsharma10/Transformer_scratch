def saffron(X, q=32, k=4, tangent_dim=1, curv_thresh=0.95, decay_rate=0.9,
            max_iter=15, verbose=False):
  '''
  SAFFRON graph construction method.

    X : (n,d)-array of coordinates
    q : int, median number of candidate friends per vertex
    k : int, number of friends to select per vertex, k < q
    tangent_dim : int, dimensionality of manifold tangent space
    curv_thresh : float, tolerance to curvature, lambda in the paper
    decay_rate : float, controls step size per iteration, between 0 and 1
    max_iter : int, cap on number of iterations
    verbose : bool, print goodness measure per iteration when True

  From "Tangent Space Guided Intelligent Neighbor Finding",
    by Gashler & Martinez, 2011.
  See http://axon.cs.byu.edu/papers/gashler2011ijcnn1.pdf
  '''
  n = len(X)
  dist = pairwise_distances(X)
  idx = np.argpartition(dist, q)[:, q]
  # radius for finding candidate friends: median distance to qth neighbor
  r = np.median(dist[np.arange(n), idx])

  # make candidate graph + weights
  W = neighbor_graph(dist, precomputed=True, epsilon=r).matrix('csr')
  # NOTE: this differs from the paper, where W.data[:] = 1 initially
  W.data[:] = 1 / W.data
  # row normalize
  normalize(W, norm='l1', axis=1, copy=False)
  # XXX: hacky densify
  W = W.toarray()

  # iterate to learn optimal weights
  prev_goodness = 1e-12
  for it in range(max_iter):
    goodness = 0
    S = _estimate_tangent_spaces(X, W, tangent_dim)
    # find aligned candidates
    for i, row in enumerate(W):
      nbrs = row.nonzero()[-1]

      # compute alignment scores
      edges = X[nbrs] - X[i]
      edge_norms = (edges**2).sum(axis=1)
      a1 = (edges.dot(S[i])**2).sum(axis=1) / edge_norms
      a2 = (np.einsum('ij,ijk->ik', edges, S[nbrs])**2).sum(axis=1) / edge_norms
      a3 = _principal_angle(S[i], S[nbrs]) ** 2
      x = (np.minimum(curv_thresh, a1) *
           np.minimum(curv_thresh, a2) *
           np.minimum(curv_thresh, a3))

      # decay weight of least-aligned candidates
      excess = x.shape[0] - k
      if excess > 0:
        bad_idx = np.argpartition(x, excess-1)[:excess]
        W[i, nbrs[bad_idx]] *= decay_rate
        W[i] /= W[i].sum()

      # update goodness measure (weighted alignment)
      goodness += x.dot(W[i,nbrs])

    if verbose:  # pragma: no cover
      goodness /= n
      print(it, goodness, goodness / prev_goodness)
    if goodness / prev_goodness <= 1.0001:
      break
    prev_goodness = goodness
  else:
    warnings.warn('Failed to converge after %d iterations.' % max_iter)

  # use the largest k weights for each row of W, weighted by original distance
  indptr, indices, data = [0], [], []
  for i, row in enumerate(W):
    nbrs = row.nonzero()[-1]
    if len(nbrs) > k:
      nbrs = nbrs[np.argpartition(row[nbrs], len(nbrs)-k)[-k:]]
    indices.extend(nbrs)
    indptr.append(len(nbrs))
    data.extend(dist[i, nbrs])
  indptr = np.cumsum(indptr)
  data = np.array(data)
  indices = np.array(indices)
  W = ss.csr_matrix((data, indices, indptr), shape=W.shape)
  return Graph.from_adj_matrix(W)