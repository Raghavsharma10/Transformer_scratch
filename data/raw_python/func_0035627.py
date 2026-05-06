def node_centroid_hill_climbing(G, relax=1, num_centerings=20, verbose=False):
  '''Iterative reordering method based on alternating rounds of node-centering
  and hill-climbing search.'''
  # Initialize order with BFS from a random start node.
  order = _breadth_first_order(G)
  for it in range(num_centerings):
    B = permute_graph(G, order).bandwidth()
    nc_order = _node_center(G, order, relax=relax)
    nc_B = permute_graph(G, nc_order).bandwidth()
    if nc_B < B:
      if verbose:  # pragma: no cover
        print('post-center', B, nc_B)
      order = nc_order
    order = _hill_climbing(G, order, verbose=verbose)
  return permute_graph(G, order)