def color_greedy(self):
    '''Returns a greedy vertex coloring as an array of ints.'''
    n = self.num_vertices()
    coloring = np.zeros(n, dtype=int)
    for i, nbrs in enumerate(self.adj_list()):
      nbr_colors = set(coloring[nbrs])
      for c in count(1):
        if c not in nbr_colors:
          coloring[i] = c
          break
    return coloring