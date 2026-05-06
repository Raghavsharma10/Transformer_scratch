def neighborhood_subgraph(self, start_idx, radius=1, weighted=True,
                            directed=True, return_mask=False):
    '''Returns a subgraph containing only vertices within a given
       geodesic radius of start_idx.'''
    adj = self.matrix('dense', 'csr', 'csc')
    dist = ssc.dijkstra(adj, directed=directed, indices=start_idx,
                        unweighted=(not weighted), limit=radius)
    mask = np.isfinite(dist)
    sub_adj = adj[mask][:,mask]
    g = self.__class__.from_adj_matrix(sub_adj)
    if return_mask:
      return g, mask
    return g