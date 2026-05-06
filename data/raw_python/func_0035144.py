def nearest_neighbors(query_pts, target_pts=None, metric='euclidean',
                      k=None, epsilon=None, return_dists=False,
                      precomputed=False):
  '''Find nearest neighbors of query points from a matrix of target points.

  Returns a list of indices of neighboring points, one list per query.
  If no target_pts are specified, distances are calculated within query_pts.
  When return_dists is True, returns two lists: (distances, indices)
  '''
  if k is None and epsilon is None:
    raise ValueError('Must provide `k` or `epsilon`.')

  # TODO: deprecate the precomputed kwarg
  precomputed = precomputed or (metric == 'precomputed')

  if precomputed and target_pts is not None:
    raise ValueError('`target_pts` cannot be used with precomputed distances')

  query_pts = np.array(query_pts)
  if len(query_pts.shape) == 1:
    query_pts = query_pts.reshape((1,-1))  # ensure that the query is a 1xD row

  if precomputed:
    dists = query_pts.copy()
  else:
    dists = pairwise_distances(query_pts, Y=target_pts, metric=metric)

  if epsilon is not None:
    if k is not None:
      # kNN filtering
      _, not_nn = _min_k_indices(dists, k, inv_ind=True)
      dists[np.arange(dists.shape[0]), not_nn.T] = np.inf
    # epsilon-ball
    is_close = dists <= epsilon
    if return_dists:
      nnis,nnds = [],[]
      for i,row in enumerate(is_close):
        nns = np.nonzero(row)[0]
        nnis.append(nns)
        nnds.append(dists[i,nns])
      return nnds, nnis
    return np.array([np.nonzero(row)[0] for row in is_close])

  # knn
  nns = _min_k_indices(dists,k)
  if return_dists:
    # index each row of dists by each row of nns
    row_inds = np.arange(len(nns))[:,np.newaxis]
    nn_dists = dists[row_inds, nns]
    return nn_dists, nns
  return nns