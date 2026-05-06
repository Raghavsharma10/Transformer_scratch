def circle_tear(self, spanning_tree='mst', cycle_len_thresh=5, spt_idx=None,
                  copy=True):
    '''Circular graph tearing.

    spanning_tree: one of {'mst', 'spt'}
    cycle_len_thresh: int, length of longest allowable cycle
    spt_idx: int, start vertex for shortest_path_subtree, random if None

    From "How to project 'circular' manifolds using geodesic distances?"
      by Lee & Verleysen, ESANN 2004.

    See also: shortest_path_subtree, minimum_spanning_subtree
    '''
    # make the initial spanning tree graph
    if spanning_tree == 'mst':
      tree = self.minimum_spanning_subtree().matrix()
    elif spanning_tree == 'spt':
      if spt_idx is None:
        spt_idx = np.random.choice(self.num_vertices())
      tree = self.shortest_path_subtree(spt_idx, directed=False).matrix()

    # find edges in self but not in the tree
    potential_edges = np.argwhere(ss.triu(self.matrix() - tree))

    # remove edges that induce large cycles
    ii, jj = _find_cycle_inducers(tree, potential_edges, cycle_len_thresh)
    return self.remove_edges(ii, jj, symmetric=True, copy=copy)