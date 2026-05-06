def delaunay_graph(X, weighted=False):
  '''Delaunay triangulation graph.
  '''
  e1, e2 = _delaunay_edges(X)
  pairs = np.column_stack((e1, e2))
  w = paired_distances(X[e1], X[e2]) if weighted else None
  return Graph.from_edge_pairs(pairs, num_vertices=X.shape[0], symmetric=True,
                               weights=w)