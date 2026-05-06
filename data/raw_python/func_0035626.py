def laplacian_reordering(G):
  '''Reorder vertices using the eigenvector of the graph Laplacian corresponding
  to the first positive eigenvalue.'''
  L = G.laplacian()
  vals, vecs = np.linalg.eigh(L)
  min_positive_idx = np.argmax(vals == vals[vals>0].min())
  vec = vecs[:, min_positive_idx]
  return permute_graph(G, np.argsort(vec))