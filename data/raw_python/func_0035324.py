def isograph(self, min_weight=None):
    '''Remove short-circuit edges using the Isograph algorithm.

    min_weight : float, optional
        Minimum weight of edges to consider removing. Defaults to max(MST).

    From "Isograph: Neighbourhood Graph Construction Based On Geodesic Distance
          For Semi-Supervised Learning" by Ghazvininejad et al., 2011.
    Note: This uses the non-iterative algorithm which removes edges
        rather than reweighting them.
    '''
    W = self.matrix('dense')
    # get candidate edges: all edges - MST edges
    tree = self.minimum_spanning_subtree()
    candidates = np.argwhere((W - tree.matrix('dense')) > 0)
    cand_weights = W[candidates[:,0], candidates[:,1]]
    # order by increasing edge weight
    order = np.argsort(cand_weights)
    cand_weights = cand_weights[order]
    # disregard edges shorter than a threshold
    if min_weight is None:
      min_weight = tree.edge_weights().max()
    idx = np.searchsorted(cand_weights, min_weight)
    cand_weights = cand_weights[idx:]
    candidates = candidates[order[idx:]]
    # check each candidate edge
    to_remove = np.zeros_like(cand_weights, dtype=bool)
    for i, (u,v) in enumerate(candidates):
      W_uv = np.where(W < cand_weights[i], W, 0)
      len_uv = ssc.dijkstra(W_uv, indices=u, unweighted=True, limit=2)[v]
      if len_uv > 2:
        to_remove[i] = True
    ii, jj = candidates[to_remove].T
    return self.remove_edges(ii, jj, copy=True)