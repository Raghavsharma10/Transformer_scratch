def directed_graph(trajectories, k=5, verbose=False, pruning_thresh=0,
                   return_coords=False):
  '''Directed graph construction alg. from Johns & Mahadevan, ICML '07.
  trajectories: list of NxD arrays of ordered states
  '''
  X = np.vstack(trajectories)
  G = neighbor_graph(X, k=k)
  if pruning_thresh > 0:
    traj_len = map(len, trajectories)
    G = _prune_edges(G, X, traj_len, pruning_thresh, verbose=verbose)
  if return_coords:
    return G, X
  return G