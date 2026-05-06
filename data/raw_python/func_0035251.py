def flesh_out(X, W, embed_dim, CC_labels, dist_mult=2.0, angle_thresh=0.2,
              min_shortcircuit=4, max_degree=5, verbose=False):
  '''Given a connected graph adj matrix (W), add edges to flesh it out.'''
  W = W.astype(bool)
  assert np.all(W == W.T), 'graph given to flesh_out must be symmetric'
  D = pairwise_distances(X, metric='sqeuclidean')

  # compute average edge lengths for each point
  avg_edge_length = np.empty(X.shape[0])
  for i,nbr_mask in enumerate(W):
    avg_edge_length[i] = D[i,nbr_mask].mean()

  # candidate edges must satisfy edge length for at least one end point
  dist_thresh = dist_mult * avg_edge_length
  dist_mask = (D < dist_thresh) | (D < dist_thresh[:,None])
  # candidate edges must connect points >= min_shortcircuit hops away
  hops_mask = np.isinf(dijkstra(W, unweighted=True, limit=min_shortcircuit-1))
  # candidate edges must not already be connected, or in the same initial CC
  CC_mask = CC_labels != CC_labels[:,None]
  candidate_edges = ~W & dist_mask & hops_mask & CC_mask
  if verbose:  # pragma: no cover
    print('before F:', candidate_edges.sum(), 'potentials')

  # calc subspaces
  subspaces, _ = cluster_subspaces(X, embed_dim, CC_labels.max()+1, CC_labels)

  # upper triangular avoids p,q <-> q,p repeats
  ii,jj = np.where(np.triu(candidate_edges))
  # Get angles
  edge_dirs = X[ii] - X[jj]
  ssi = subspaces[CC_labels[ii]]
  ssj = subspaces[CC_labels[jj]]
  F = edge_cluster_angle(edge_dirs, ssi, ssj)

  mask = F < angle_thresh
  edge_ii = ii[mask]
  edge_jj = jj[mask]
  edge_order = np.argsort(F[mask])
  if verbose:  # pragma: no cover
    print('got', len(edge_ii), 'potential edges')
  # Prevent any one node from getting a really high degree
  degree = W.sum(axis=0)
  sorted_edges = np.column_stack((edge_ii, edge_jj))[edge_order]
  for e in sorted_edges:
    if degree[e].max() < max_degree:
      W[e[0],e[1]] = True
      W[e[1],e[0]] = True
      degree[e] += 1
  return Graph.from_adj_matrix(np.where(W, np.sqrt(D), 0))