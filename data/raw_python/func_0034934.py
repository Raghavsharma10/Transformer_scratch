def urquhart_graph(X, weighted=False):
  '''Urquhart graph: made from the 2 shortest edges of each Delaunay triangle.
  '''
  e1, e2 = _delaunay_edges(X)
  w = paired_distances(X[e1], X[e2])
  mask = np.ones_like(w, dtype=bool)
  bad_inds = w.reshape((-1, 3)).argmax(axis=1) + np.arange(0, len(e1), 3)
  mask[bad_inds] = False

  weights = w[mask] if weighted else None
  pairs = np.column_stack((e1[mask], e2[mask]))
  return Graph.from_edge_pairs(pairs, num_vertices=X.shape[0], symmetric=True,
                               weights=weights)