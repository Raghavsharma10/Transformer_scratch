def permute_graph(G, order):
  '''Reorder the graph's vertices, returning a copy of the input graph.
  order : integer array-like, some permutation of range(G.num_vertices()).
  '''
  adj = G.matrix('dense')
  adj = adj[np.ix_(order, order)]
  return Graph.from_adj_matrix(adj)